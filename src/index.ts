#!/usr/bin/env node
import { AgentSideConnection, ndJsonStream } from "@agentclientprotocol/sdk";
import { CursorAgent } from "./agent.js";

const input = new WritableStream<Uint8Array>({
  write(chunk) {
    return new Promise<void>((resolve, reject) => {
      process.stdout.write(chunk, (err) => (err ? reject(err) : resolve()));
    });
  },
});

const output = new ReadableStream<Uint8Array>({
  start(controller) {
    process.stdin.on("data", (chunk: Buffer) => controller.enqueue(new Uint8Array(chunk)));
    process.stdin.on("end", () => controller.close());
    process.stdin.on("error", (err) => controller.error(err));
  },
});

const stream = ndJsonStream(input, output);
new AgentSideConnection((conn) => new CursorAgent(conn), stream);

process.stdin.resume();
await new Promise<void>((resolve, reject) => {
  process.stdin.on("end", () => resolve());
  process.stdin.on("error", (error) => reject(error));
});
