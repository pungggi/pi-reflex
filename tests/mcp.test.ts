import { describe, expect, it } from "vitest";
import { Engine } from "../src/engine/engine.js";
import type { SystemOneResult } from "../src/core/types.js";
import { createToolCores } from "../src/extension/cores.js";
import { ReflexMcpServer } from "../src/mcp/server.js";

function fakeLoader(): () => Promise<Engine> {
  return async () =>
    ({
      name: "fake",
      async systemOne(_state: unknown, questions: Parameters<Engine["systemOne"]>[1]): Promise<SystemOneResult> {
        const answers: SystemOneResult["answers"] = {};
        for (const [qid, q] of Object.entries(questions)) {
          if (q.type === "choice") {
            (answers as Record<string, unknown>)[qid] = { type: "choice", choice: "billing", probabilities: { billing: 0.9 }, confidence: 0.9, action: { act_probability: 0.5 } };
          } else if (q.type === "noul") {
            (answers as Record<string, unknown>)[qid] = { type: "noul", noul: 0.42, confidence: 0.8, action: { act_probability: 0.5 } };
          } else {
            (answers as Record<string, unknown>)[qid] = { type: "score", score: 1.0, legend: {}, probabilities: { "0": 0.5, "1": 0.5 }, confidence: 0.5, action: { act_probability: 0.5 } };
          }
        }
        return { model: "fake", answers, usage: { input_tokens: 3, output_tokens: 0 } };
      },
      async batchQuestion() {
        return [];
      },
    }) as unknown as Engine;
}

function makeServer(): ReflexMcpServer {
  return new ReflexMcpServer(createToolCores(fakeLoader()));
}

async function call(server: ReflexMcpServer, obj: unknown): Promise<any> {
  const res = await server.handleLine(JSON.stringify(obj));
  expect(res).not.toBeNull();
  return JSON.parse(res!);
}

describe("MCP stdio server", () => {
  it("initialize echoes the client protocol version", async () => {
    const res = await call(makeServer(), {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2025-06-18" },
    });
    expect(res.result.protocolVersion).toBe("2025-06-18");
    expect(res.result.serverInfo.name).toBe("pi-reflex");
    expect(res.result.capabilities.tools).toEqual({});
  });

  it("notifications return null (no response)", async () => {
    expect(await makeServer().handleLine(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }))).toBeNull();
  });

  it("answers ping", async () => {
    const res = await call(makeServer(), { jsonrpc: "2.0", id: 2, method: "ping" });
    expect(res.result).toEqual({});
  });

  it("tools/list exposes the four reflex tools with JSON Schema", async () => {
    const res = await call(makeServer(), { jsonrpc: "2.0", id: 3, method: "tools/list" });
    const names = res.result.tools.map((t: { name: string }) => t.name);
    expect(names).toEqual(["reflex_decide", "reflex_judge", "reflex_rate", "reflex_route"]);
    for (const t of res.result.tools) {
      expect(t.inputSchema.type).toBe("object");
      expect(t.inputSchema.required.length).toBeGreaterThan(0);
    }
  });

  it("tools/call runs the judge core and returns text + structuredContent", async () => {
    const res = await call(makeServer(), {
      jsonrpc: "2.0",
      id: 4,
      method: "tools/call",
      params: { name: "reflex_judge", arguments: { state: "prod is down", question: "Urgent?" } },
    });
    expect(res.result.content[0].text).toContain("P(true)=0.42");
    expect(res.result.structuredContent).toMatchObject({ type: "bool", probability: 0.42 });
    expect(res.result.isError).toBeUndefined();
  });

  it("tools/call unknown tool → -32602; tool errors → isError result", async () => {
    const server = makeServer();
    const unknown = await call(server, { jsonrpc: "2.0", id: 5, method: "tools/call", params: { name: "nope", arguments: {} } });
    expect(unknown.error.code).toBe(-32602);
    const bad = await call(server, {
      jsonrpc: "2.0",
      id: 6,
      method: "tools/call",
      params: { name: "reflex_rate", arguments: { state: "x", instructions: "y", levels: ["only"] } },
    });
    expect(bad.result.isError).toBe(true);
    expect(bad.result.content[0].text).toMatch(/at least 2 levels/);
  });

  it("unknown methods → -32601; malformed lines → -32700", async () => {
    const server = makeServer();
    const unknown = await call(server, { jsonrpc: "2.0", id: 7, method: "resources/list" });
    expect(unknown.error.code).toBe(-32601);
    const parse = JSON.parse((await server.handleLine("not json"))!);
    expect(parse.error.code).toBe(-32700);
  });
});
