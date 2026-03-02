import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

export type CursorModel = {
  modelId: string;
  name: string;
};

export type CursorPromptResult = {
  text: string;
  sessionId?: string;
  usage?: unknown;
};

export type CursorStreamEvent = Record<string, unknown>;

type RunCommandResult = {
  stdout: string;
  stderr: string;
  exitCode: number;
};

export class CursorBridge {
  async getStatus(): Promise<{ authenticated: boolean; raw: string }> {
    const { stdout } = await this.runCursorCommand(["status"]);
    const raw = stdout.trim();
    return {
      raw,
      authenticated: this.parseAuthStatus(raw),
    };
  }

  async listModels(): Promise<CursorModel[]> {
    const { stdout } = await this.runCursorCommand(["models"]);
    return stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const [modelId, ...rest] = line.split(" - ");
        return {
          modelId: modelId.trim(),
          name: rest.join(" - ").trim() || modelId.trim(),
        };
      })
      .filter((entry) => entry.modelId.length > 0);
  }

  async createChat(): Promise<string> {
    const { stdout } = await this.runCursorCommand(["create-chat"]);
    const chatId = stdout.trim();
    if (!chatId) {
      throw new Error("cursor-agent create-chat returned an empty chat ID");
    }
    return chatId;
  }

  async streamPrompt(params: {
    cwd: string;
    prompt: string;
    chatId?: string;
    model?: string;
    signal?: AbortSignal;
    onChunk?: (event: CursorStreamEvent) => Promise<void> | void;
  }): Promise<CursorPromptResult> {
    const args = [
      "--print",
      "--output-format",
      "stream-json",
      "--stream-partial-output",
      "--force",
      "--trust",
      "--workspace",
      params.cwd,
    ];

    if (params.model) {
      args.push("--model", params.model);
    }
    if (params.chatId) {
      args.push("--resume", params.chatId);
    }
    args.push(params.prompt);

    const child = spawn("cursor-agent", args, {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stderr = "";
    let finalResult: CursorPromptResult | undefined;

    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    const onAbort = () => {
      if (child.killed) {
        return;
      }
      child.kill("SIGTERM");
      const timeout = setTimeout(() => {
        if (!child.killed) {
          child.kill("SIGKILL");
        }
      }, 500);
      timeout.unref();
    };

    if (params.signal?.aborted) {
      onAbort();
    } else if (params.signal) {
      params.signal.addEventListener("abort", onAbort, { once: true });
    }

    const closePromise = new Promise<RunCommandResult>((resolve, reject) => {
      child.on("error", reject);
      child.on("close", (exitCode) => {
        resolve({
          stdout: "",
          stderr,
          exitCode: exitCode ?? -1,
        });
      });
    });

    try {
      const rl = createInterface({ input: child.stdout });
      for await (const line of rl) {
        if (!line.trim()) {
          continue;
        }
        let event: CursorStreamEvent;
        try {
          event = JSON.parse(line) as CursorStreamEvent;
        } catch {
          continue;
        }

        await params.onChunk?.(event);

        if (event.type === "result") {
          if (event.is_error === true) {
            throw new Error(typeof event.result === "string" ? event.result : "cursor-agent returned an error result");
          }
          finalResult = {
            text: typeof event.result === "string" ? event.result : "",
            sessionId: typeof event.session_id === "string" ? event.session_id : undefined,
            usage: event.usage,
          };
        }
      }

      const { exitCode, stderr: stderrText } = await closePromise;
      if (params.signal?.aborted) {
        throw new Error("aborted");
      }
      if (exitCode !== 0 && !finalResult) {
        throw new Error(`cursor-agent exited with code ${exitCode}: ${stderrText.trim() || "unknown error"}`);
      }
      if (!finalResult) {
        throw new Error("cursor-agent stream ended without a result event");
      }
      return finalResult;
    } finally {
      params.signal?.removeEventListener("abort", onAbort);
    }
  }

  private async runCursorCommand(args: string[]): Promise<RunCommandResult> {
    const child = spawn("cursor-agent", args, {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");

    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    const result = await new Promise<RunCommandResult>((resolve, reject) => {
      child.on("error", reject);
      child.on("close", (exitCode) => {
        resolve({
          stdout,
          stderr,
          exitCode: exitCode ?? -1,
        });
      });
    });

    if (result.exitCode !== 0) {
      throw new Error(`cursor-agent ${args.join(" ")} failed: ${result.stderr.trim() || `exit code ${result.exitCode}`}`);
    }

    return result;
  }

  private parseAuthStatus(raw: string): boolean {
    const normalized = raw.toLowerCase();
    if (!normalized) {
      return false;
    }
    if (normalized.includes("not logged in") || normalized.includes("unauthenticated")) {
      return false;
    }
    if (normalized.includes("logged in") || normalized.includes("authenticated")) {
      return true;
    }
    return true;
  }
}
