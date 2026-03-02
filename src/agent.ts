import { randomUUID } from "node:crypto";
import {
  PROTOCOL_VERSION,
  type Agent,
  type AgentSideConnection,
  type AuthenticateRequest,
  type AuthenticateResponse,
  type CancelNotification,
  type InitializeRequest,
  type InitializeResponse,
  type ListSessionsRequest,
  type ListSessionsResponse,
  type LoadSessionRequest,
  type LoadSessionResponse,
  type NewSessionRequest,
  type NewSessionResponse,
  type PromptRequest,
  type PromptResponse,
  type ResumeSessionRequest,
  type ResumeSessionResponse,
  type SetSessionModeRequest,
  type SetSessionModeResponse,
  type SetSessionModelRequest,
  type SetSessionModelResponse,
} from "@agentclientprotocol/sdk";
import { CursorBridge, type CursorModel, type CursorStreamEvent } from "./cursor.js";
import { SessionStore } from "./session.js";

export class CursorAgent implements Agent {
  private readonly pendingPrompts = new Map<string, AbortController>();
  private models: CursorModel[] = [];

  constructor(
    private readonly conn: AgentSideConnection,
    private readonly cursor = new CursorBridge(),
    private readonly sessions = new SessionStore(),
  ) {}

  async initialize(_params: InitializeRequest): Promise<InitializeResponse> {
    const [status, models] = await Promise.all([this.cursor.getStatus(), this.cursor.listModels()]);
    this.models = models;

    return {
      protocolVersion: PROTOCOL_VERSION,
      agentInfo: {
        name: "cursor-cli-acp",
        version: "0.1.2",
      },
      agentCapabilities: {
        loadSession: true,
        sessionCapabilities: {
          list: {},
          resume: {},
        },
      },
      _meta: {
        cursorAgent: {
          authenticated: status.authenticated,
          status: status.raw,
          models: models.map((model) => ({
            modelId: model.modelId,
            name: model.name,
          })),
        },
      },
    };
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    const chatId = await this.cursor.createChat();
    const sessionId = randomUUID();

    const defaultModel = this.models[0]?.modelId;
    this.sessions.create(sessionId, {
      chatId,
      cwd: params.cwd,
      model: defaultModel,
    });

    return {
      sessionId,
      models: this.getModelState(defaultModel),
    };
  }

  async loadSession(params: LoadSessionRequest): Promise<LoadSessionResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw new Error(`Session not found in memory: ${params.sessionId}`);
    }
    return {
      models: this.getModelState(session.model),
    };
  }

  async authenticate(_params: AuthenticateRequest): Promise<AuthenticateResponse | void> {
    return {};
  }

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }

    this.pendingPrompts.get(params.sessionId)?.abort();
    const abortController = new AbortController();
    this.pendingPrompts.set(params.sessionId, abortController);

    const text = params.prompt
      .filter((block): block is { type: "text"; text: string } => block.type === "text")
      .map((block) => block.text)
      .join("\n");

    const startedToolCalls = new Set<string>();

    try {
      await this.cursor.streamPrompt({
        cwd: session.cwd,
        chatId: session.chatId,
        model: session.model,
        prompt: text,
        signal: abortController.signal,
        onChunk: async (event) => {
          if (abortController.signal.aborted) {
            throw new Error("aborted");
          }
          await this.forwardChunk(params.sessionId, event, startedToolCalls);
        },
      });

      this.sessions.update(params.sessionId, {});
      return { stopReason: "end_turn" };
    } catch (error) {
      if (abortController.signal.aborted || isAbortError(error)) {
        return { stopReason: "cancelled" };
      }
      throw error;
    } finally {
      const controller = this.pendingPrompts.get(params.sessionId);
      if (controller === abortController) {
        this.pendingPrompts.delete(params.sessionId);
      }
    }
  }

  async cancel(params: CancelNotification): Promise<void> {
    this.pendingPrompts.get(params.sessionId)?.abort();
  }

  async setSessionMode(_params: SetSessionModeRequest): Promise<SetSessionModeResponse | void> {
    return {};
  }

  async unstable_setSessionModel(params: SetSessionModelRequest): Promise<SetSessionModelResponse | void> {
    if (!this.sessions.has(params.sessionId)) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }
    this.sessions.update(params.sessionId, { model: params.modelId });
    return {};
  }

  async unstable_listSessions(_params: ListSessionsRequest): Promise<ListSessionsResponse> {
    return {
      sessions: this.sessions.list().map(({ sessionId, session }) => ({
        sessionId,
        cwd: session.cwd,
        title: `Cursor chat ${session.chatId.slice(0, 8)}`,
        updatedAt: session.updatedAt,
      })),
    };
  }

  async unstable_resumeSession(params: ResumeSessionRequest): Promise<ResumeSessionResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw new Error(`Session not found in memory: ${params.sessionId}`);
    }
    return {
      models: this.getModelState(session.model),
    };
  }

  private async forwardChunk(sessionId: string, event: CursorStreamEvent, startedToolCalls: Set<string>): Promise<void> {
    const type = readString(event, "type");
    if (!type) {
      return;
    }

    if (type === "thinking") {
      const thought = readString(event, "text");
      if (thought) {
        await this.conn.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_thought_chunk",
            content: {
              type: "text",
              text: thought,
            },
          },
        });
      }
      return;
    }

    if (type === "assistant") {
      // cursor-agent sends each assistant message twice: once with timestamp_ms (streaming)
      // and once without (final accumulated duplicate). Skip the duplicate.
      if (!("timestamp_ms" in event)) {
        return;
      }
      for (const chunk of getAssistantTextChunks(event)) {
        if (!chunk) {
          continue;
        }
        await this.conn.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: {
              type: "text",
              text: chunk,
            },
          },
        });
      }
      return;
    }

    if (type === "tool_call") {
      const subtype = readString(event, "subtype");
      const toolCallId = getToolCallId(event);
      const title = getToolCallTitle(event);

      if (subtype === "started") {
        startedToolCalls.add(toolCallId);
        await this.conn.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "tool_call",
            toolCallId,
            title,
            kind: "execute",
            status: "in_progress",
            rawInput: event,
          },
        });
        return;
      }

      if (subtype === "completed" && startedToolCalls.has(toolCallId)) {
        await this.conn.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId,
            status: "completed",
            rawOutput: event,
          },
        });
      }
    }
  }

  private getModelState(currentModelId?: string) {
    if (this.models.length === 0) {
      return undefined;
    }
    const selectedModel = currentModelId ?? this.models[0]?.modelId;
    if (!selectedModel) {
      return undefined;
    }
    return {
      currentModelId: selectedModel,
      availableModels: this.models.map((model) => ({
        modelId: model.modelId,
        name: model.name,
      })),
    };
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(obj: Record<string, unknown>, key: string): string | undefined {
  const value = obj[key];
  return typeof value === "string" ? value : undefined;
}

function getAssistantTextChunks(event: CursorStreamEvent): string[] {
  const message = isRecord(event.message) ? event.message : undefined;
  if (!message) {
    return [];
  }

  const content = message.content;
  if (!Array.isArray(content)) {
    return [];
  }

  const chunks: string[] = [];
  for (const item of content) {
    if (!isRecord(item)) {
      continue;
    }
    if (item.type !== "text" || typeof item.text !== "string") {
      continue;
    }
    chunks.push(item.text);
  }
  return chunks;
}

function getToolCallId(event: CursorStreamEvent): string {
  const knownId =
    readString(event, "tool_call_id") ??
    readString(event, "call_id") ??
    readString(event, "id");
  if (knownId) {
    return knownId;
  }
  return randomUUID();
}

function getToolCallTitle(event: CursorStreamEvent): string {
  const directCommand = readString(event, "command");
  if (directCommand) {
    return `Running: ${directCommand}`;
  }

  // cursor-agent structure: event.tool_call.shellToolCall.args.command
  const toolCall = isRecord(event.tool_call) ? event.tool_call : undefined;
  if (toolCall) {
    const shellToolCall = isRecord(toolCall.shellToolCall) ? toolCall.shellToolCall : undefined;
    if (shellToolCall) {
      const args = isRecord(shellToolCall.args) ? shellToolCall.args : undefined;
      const command = args ? readString(args, "command") : undefined;
      if (command) {
        return `Running: ${command}`;
      }
      const description = readString(shellToolCall, "description");
      if (description) {
        return description;
      }
    }
  }

  return "Running cursor tool call";
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.message.toLowerCase().includes("abort");
}
