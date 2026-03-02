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
  type ToolCall,
  type ToolCallContent,
  type ToolCallLocation,
  type ToolCallUpdate,
  type ToolKind,
  type ToolCallStatus,
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
        version: "0.2.0",
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
      const parsed = parseToolCall(event);

      if (subtype === "started") {
        startedToolCalls.add(toolCallId);

        const update: ToolCall & { sessionUpdate: "tool_call" } = {
          sessionUpdate: "tool_call",
          toolCallId,
          title: parsed.title,
          kind: parsed.kind,
          status: "in_progress",
          rawInput: event,
        };

        if (parsed.locations.length > 0) {
          update.locations = parsed.locations;
        }

        await this.conn.sessionUpdate({ sessionId, update });
        return;
      }

      if (subtype === "completed" && startedToolCalls.has(toolCallId)) {
        const update: ToolCallUpdate & { sessionUpdate: "tool_call_update" } = {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: parsed.status,
          rawOutput: event,
        };

        if (parsed.content.length > 0) {
          update.content = parsed.content;
        }
        if (parsed.locations.length > 0) {
          update.locations = parsed.locations;
        }
        // Update title on completion if we got a better one (e.g. failed status)
        if (parsed.status === "failed") {
          update.title = parsed.title;
        }

        await this.conn.sessionUpdate({ sessionId, update });
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

// ─── Utility helpers ───

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(obj: Record<string, unknown>, key: string): string | undefined {
  const value = obj[key];
  return typeof value === "string" ? value : undefined;
}

function readNumber(obj: Record<string, unknown>, key: string): number | undefined {
  const value = obj[key];
  return typeof value === "number" ? value : undefined;
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

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.message.toLowerCase().includes("abort");
}

// ─── Cursor tool_call structure ───
//
// event.tool_call is a record with exactly ONE key naming the tool type:
//   readToolCall, editToolCall, shellToolCall, grepToolCall, globToolCall,
//   listDirToolCall, codebaseSearchToolCall, createFileToolCall, deleteFileToolCall, etc.
//
// That key maps to an object with:
//   - args: Record<string, unknown>  (input parameters)
//   - result?: { success?: Record<string, unknown> } | { error?: string }  (on completed events)
//   - description?: string  (human-readable, mainly on shellToolCall)

type ParsedToolCall = {
  kind: ToolKind;
  title: string;
  status: ToolCallStatus;
  content: ToolCallContent[];
  locations: ToolCallLocation[];
};

/** Extract the single tool type entry from event.tool_call */
function getToolEntry(event: CursorStreamEvent): { toolType: string; entry: Record<string, unknown> } | undefined {
  const toolCall = isRecord(event.tool_call) ? event.tool_call : undefined;
  if (!toolCall) {
    return undefined;
  }

  for (const key of Object.keys(toolCall)) {
    const entry = toolCall[key];
    if (isRecord(entry)) {
      return { toolType: key, entry };
    }
  }
  return undefined;
}

/** Get the args sub-object from a tool entry */
function getArgs(entry: Record<string, unknown>): Record<string, unknown> | undefined {
  return isRecord(entry.args) ? entry.args : undefined;
}

/** Get the result.success sub-object from a tool entry */
function getSuccess(entry: Record<string, unknown>): Record<string, unknown> | undefined {
  const result = isRecord(entry.result) ? entry.result : undefined;
  if (!result) return undefined;
  return isRecord(result.success) ? result.success : undefined;
}

/** Check if the tool result indicates failure */
function hasError(entry: Record<string, unknown>): string | undefined {
  const result = isRecord(entry.result) ? entry.result : undefined;
  if (!result) return undefined;
  if (typeof result.error === "string") return result.error;
  if (isRecord(result.error) && typeof result.error.message === "string") return result.error.message;
  // Some tools use isBackground with error
  return undefined;
}

/** Shorten an absolute file path for display */
function shortPath(path: string): string {
  // Show just the filename or last 2 segments
  const parts = path.split("/").filter(Boolean);
  if (parts.length <= 2) return path;
  return parts.slice(-2).join("/");
}

/** Main parser: inspects the cursor event and returns structured ACP data */
function parseToolCall(event: CursorStreamEvent): ParsedToolCall {
  const subtype = readString(event, "subtype");
  const toolInfo = getToolEntry(event);

  if (!toolInfo) {
    return {
      kind: "other",
      title: "Tool call",
      status: subtype === "completed" ? "completed" : "in_progress",
      content: [],
      locations: [],
    };
  }

  const { toolType, entry } = toolInfo;
  const args = getArgs(entry);
  const success = getSuccess(entry);
  const error = hasError(entry);

  if (error) {
    return {
      kind: toolTypeToKind(toolType),
      title: `Failed: ${error.slice(0, 100)}`,
      status: "failed",
      content: [{ type: "content", content: { type: "text", text: error } }],
      locations: [],
    };
  }

  switch (toolType) {
    case "readToolCall":
      return parseReadToolCall(args, success, subtype);
    case "editToolCall":
      return parseEditToolCall(args, success, subtype);
    case "shellToolCall":
      return parseShellToolCall(entry, args, success, subtype);
    case "grepToolCall":
      return parseGrepToolCall(args, success, subtype);
    case "globToolCall":
      return parseGlobToolCall(args, success, subtype);
    case "listDirToolCall":
      return parseListDirToolCall(args, success, subtype);
    case "codebaseSearchToolCall":
      return parseCodebaseSearchToolCall(args, success, subtype);
    case "createFileToolCall":
      return parseCreateFileToolCall(args, success, subtype);
    case "deleteFileToolCall":
      return parseDeleteFileToolCall(args, success, subtype);
    default:
      return parseFallbackToolCall(toolType, entry, args, success, subtype);
  }
}

function toolTypeToKind(toolType: string): ToolKind {
  switch (toolType) {
    case "readToolCall":
    case "listDirToolCall":
      return "read";
    case "editToolCall":
    case "createFileToolCall":
      return "edit";
    case "deleteFileToolCall":
      return "delete";
    case "shellToolCall":
      return "execute";
    case "grepToolCall":
    case "globToolCall":
    case "codebaseSearchToolCall":
      return "search";
    default:
      return "other";
  }
}

// ─── Per-tool-type parsers ───

function parseReadToolCall(
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const path = (args && readString(args, "path")) ?? (success && readString(success, "path")) ?? "";
  const locations: ToolCallLocation[] = path ? [{ path }] : [];
  const content: ToolCallContent[] = [];

  if (subtype === "completed" && success) {
    const fileContent = readString(success, "content");
    if (fileContent) {
      content.push({
        type: "content",
        content: { type: "text", text: fileContent },
      });
    } else {
      // Some reads return metadata only (contentBlobId) without inline content
      const totalLines = readNumber(success, "totalLines");
      const fileSize = readNumber(success, "fileSize");
      const readRange = isRecord(success.readRange) ? success.readRange : undefined;
      const startLine = readRange ? readNumber(readRange, "startLine") : undefined;
      const endLine = readRange ? readNumber(readRange, "endLine") : undefined;

      let summary = `Read ${shortPath(path)}`;
      if (totalLines !== undefined) summary += ` (${totalLines} lines, ${fileSize ?? "?"} bytes)`;
      if (startLine !== undefined && endLine !== undefined) summary += ` lines ${startLine}-${endLine}`;

      content.push({
        type: "content",
        content: { type: "text", text: summary },
      });
    }

    if (readString(success, "path")) {
      locations[0] = { path: readString(success, "path")! };
    }
  }

  return {
    kind: "read",
    title: `Read ${shortPath(path)}`,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations,
  };
}

function parseEditToolCall(
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const path = (args && readString(args, "path")) ?? (success && readString(success, "path")) ?? "";
  const locations: ToolCallLocation[] = path ? [{ path }] : [];
  const content: ToolCallContent[] = [];

  if (subtype === "completed" && success) {
    const oldText = readString(success, "beforeFullFileContent");
    const newText = readString(success, "afterFullFileContent");

    if (newText !== undefined) {
      content.push({
        type: "diff",
        path,
        newText,
        ...(oldText !== undefined ? { oldText } : {}),
      });
    } else {
      // Fallback: use the diffString if full content not available
      const diffString = readString(success, "diffString");
      if (diffString) {
        content.push({
          type: "content",
          content: { type: "text", text: diffString },
        });
      }
    }
  }

  return {
    kind: "edit",
    title: `Edit ${shortPath(path)}`,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations,
  };
}

function parseShellToolCall(
  entry: Record<string, unknown>,
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const command = (args && readString(args, "command")) ?? "";
  const description = readString(entry, "description") ?? (args && readString(args, "description"));
  const content: ToolCallContent[] = [];
  let status: ToolCallStatus = subtype === "completed" ? "completed" : "in_progress";

  if (subtype === "completed" && success) {
    const exitCode = readNumber(success, "exitCode");
    const stdout = readString(success, "stdout") ?? "";
    const stderr = readString(success, "stderr") ?? "";
    const interleaved = readString(success, "interleavedOutput");

    // Use interleaved output if available, otherwise combine stdout+stderr
    const output = interleaved ?? [stdout, stderr].filter(Boolean).join("\n");

    if (output) {
      content.push({
        type: "content",
        content: { type: "text", text: output },
      });
    }

    if (exitCode !== undefined && exitCode !== 0) {
      status = "failed";
    }
  }

  const title = command
    ? `Run \`${command.length > 80 ? command.slice(0, 77) + "..." : command}\``
    : description ?? "Run command";

  return {
    kind: "execute",
    title,
    status,
    content,
    locations: [],
  };
}

function parseGrepToolCall(
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const pattern = (args && readString(args, "pattern")) ?? "";
  const content: ToolCallContent[] = [];
  const locations: ToolCallLocation[] = [];

  if (subtype === "completed" && success) {
    const workspaceResults = isRecord(success.workspaceResults) ? success.workspaceResults : undefined;
    if (workspaceResults) {
      const lines: string[] = [];
      for (const wsKey of Object.keys(workspaceResults)) {
        const wsResult = isRecord(workspaceResults[wsKey]) ? workspaceResults[wsKey] : undefined;
        const resultContent = wsResult && isRecord(wsResult.content) ? wsResult.content : undefined;
        const matches = resultContent && Array.isArray(resultContent.matches) ? resultContent.matches : [];

        for (const fileMatch of matches) {
          if (!isRecord(fileMatch)) continue;
          const file = readString(fileMatch, "file") ?? "";
          const fileMatches = Array.isArray(fileMatch.matches) ? fileMatch.matches : [];

          for (const match of fileMatches) {
            if (!isRecord(match)) continue;
            const lineNumber = readNumber(match, "lineNumber");
            const matchContent = readString(match, "content") ?? "";
            const isContext = match.isContextLine === true;

            if (!isContext) {
              lines.push(`${file}:${lineNumber ?? "?"}: ${matchContent}`);
              if (file) {
                locations.push({ path: file, ...(lineNumber !== undefined ? { line: lineNumber } : {}) });
              }
            }
          }
        }
      }

      if (lines.length > 0) {
        content.push({
          type: "content",
          content: { type: "text", text: lines.join("\n") },
        });
      } else {
        content.push({
          type: "content",
          content: { type: "text", text: "No matches found" },
        });
      }
    }
  }

  return {
    kind: "search",
    title: `Search \`${pattern}\``,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations,
  };
}

function parseGlobToolCall(
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const targetDir = (args && readString(args, "targetDirectory")) ?? "";
  const globPattern = (args && readString(args, "globPattern")) ?? "*";
  const content: ToolCallContent[] = [];
  const locations: ToolCallLocation[] = targetDir ? [{ path: targetDir }] : [];

  if (subtype === "completed" && success) {
    const files = Array.isArray(success.files) ? success.files : [];
    const totalFiles = readNumber(success, "totalFiles");

    if (files.length > 0) {
      const fileList = files.filter((f): f is string => typeof f === "string").join("\n");
      let text = fileList;
      if (totalFiles !== undefined && totalFiles > files.length) {
        text += `\n... and ${totalFiles - files.length} more`;
      }
      content.push({
        type: "content",
        content: { type: "text", text },
      });
    } else {
      content.push({
        type: "content",
        content: { type: "text", text: "No files found" },
      });
    }
  }

  return {
    kind: "search",
    title: `List ${shortPath(targetDir)}/${globPattern}`,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations,
  };
}

function parseListDirToolCall(
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const path = (args && readString(args, "path")) ?? "";
  const content: ToolCallContent[] = [];

  if (subtype === "completed" && success) {
    const entries = Array.isArray(success.entries) ? success.entries : [];
    if (entries.length > 0) {
      const listing = entries.filter((e): e is string => typeof e === "string").join("\n");
      content.push({
        type: "content",
        content: { type: "text", text: listing },
      });
    }
  }

  return {
    kind: "read",
    title: `List ${shortPath(path)}`,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations: path ? [{ path }] : [],
  };
}

function parseCodebaseSearchToolCall(
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const query = (args && readString(args, "query")) ?? (args && readString(args, "pattern")) ?? "";
  const content: ToolCallContent[] = [];
  const locations: ToolCallLocation[] = [];

  if (subtype === "completed" && success) {
    const results = Array.isArray(success.results) ? success.results : [];
    const lines: string[] = [];
    for (const result of results) {
      if (!isRecord(result)) continue;
      const file = readString(result, "file") ?? readString(result, "path") ?? "";
      const line = readNumber(result, "line") ?? readNumber(result, "lineNumber");
      const snippet = readString(result, "content") ?? readString(result, "snippet") ?? "";

      if (file) {
        lines.push(`${file}${line !== undefined ? `:${line}` : ""}: ${snippet}`);
        locations.push({ path: file, ...(line !== undefined ? { line } : {}) });
      }
    }

    if (lines.length > 0) {
      content.push({
        type: "content",
        content: { type: "text", text: lines.join("\n") },
      });
    }
  }

  return {
    kind: "search",
    title: `Search \`${query}\``,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations,
  };
}

function parseCreateFileToolCall(
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const path = (args && readString(args, "path")) ?? (success && readString(success, "path")) ?? "";
  const content: ToolCallContent[] = [];

  if (subtype === "completed" && success) {
    const newText = readString(success, "afterFullFileContent") ?? readString(args ?? {}, "content") ?? "";
    if (newText) {
      content.push({
        type: "diff",
        path,
        newText,
        oldText: "",
      });
    }
  }

  return {
    kind: "edit",
    title: `Create ${shortPath(path)}`,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations: path ? [{ path }] : [],
  };
}

function parseDeleteFileToolCall(
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  const path = (args && readString(args, "path")) ?? "";
  const content: ToolCallContent[] = [];

  if (subtype === "completed") {
    content.push({
      type: "content",
      content: { type: "text", text: `Deleted ${path}` },
    });
  }

  return {
    kind: "delete",
    title: `Delete ${shortPath(path)}`,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations: path ? [{ path }] : [],
  };
}

function parseFallbackToolCall(
  toolType: string,
  entry: Record<string, unknown>,
  args: Record<string, unknown> | undefined,
  success: Record<string, unknown> | undefined,
  subtype: string | undefined,
): ParsedToolCall {
  // Try to extract something useful from unknown tool types
  const description = readString(entry, "description");
  const content: ToolCallContent[] = [];
  const locations: ToolCallLocation[] = [];

  // Try to find a path in the args
  const path = args && readString(args, "path");
  if (path) {
    locations.push({ path });
  }

  if (subtype === "completed" && success) {
    // Dump any text-like result
    const text = readString(success, "content") ?? readString(success, "output") ?? readString(success, "result");
    if (text) {
      content.push({
        type: "content",
        content: { type: "text", text },
      });
    }
  }

  // Format the tool type for display: "shellToolCall" -> "Shell tool call"
  const displayName = toolType
    .replace(/ToolCall$/, "")
    .replace(/([A-Z])/g, " $1")
    .trim()
    .toLowerCase();

  return {
    kind: "other",
    title: description ?? `${displayName.charAt(0).toUpperCase() + displayName.slice(1)}`,
    status: subtype === "completed" ? "completed" : "in_progress",
    content,
    locations,
  };
}
