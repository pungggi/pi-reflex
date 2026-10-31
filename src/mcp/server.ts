/**
 * Minimal zero-dependency MCP stdio server exposing the pi-reflex tools to any
 * MCP client (pi, Claude Code, Cursor, …). One JSON-RPC message per line.
 *
 * Wire it up manually:
 *   pi mcp add reflex -- node <pkg>/bin/pi-reflex-mcp.js
 * or let the pi extension register it with PI_REFLEX_MCP=1.
 *
 * Kept free of pi/typebox imports: it runs standalone with only the pi-reflex
 * runtime (engine + presets + tool cores).
 */
import { createInterface } from "node:readline";
import { createRequire } from "node:module";
import { Engine } from "../engine/engine.js";
import { createToolCores, type ToolCores } from "../extension/cores.js";
import { ensureEngine, findLocalEngine, type EngineName, type Quant } from "../engine/download.js";

const PROTOCOL_VERSION = "2024-11-05";
/** MCP revisions this server implements (tools only — stable across these). */
const SUPPORTED_PROTOCOL_VERSIONS = new Set(["2024-11-05", "2025-03-26", "2025-06-18"]);
// Resolve from the package root (../../package.json from both src/mcp and
// dist/mcp) so serverInfo.version always matches package.json — a hardcoded
// copy drifted at 0.0.1 for four releases.
const { version: PKG_VERSION } = createRequire(import.meta.url)("../../package.json") as { version: string };
const SERVER_INFO = { name: "pi-reflex", version: PKG_VERSION };

/**
 * Hand-written JSON Schema mirrors of src/extension/schemas.ts (the typebox
 * source of truth). tests/mcp.test.ts asserts deep equality with the typebox
 * definitions after a JSON round-trip, so the two surfaces cannot drift; the
 * mirrors keep this server free of pi/typebox imports. outputSchema is declared
 * per the 2025-06-18 MCP revision (older clients ignore it).
 */
const TOOL_SPECS = [
  {
    name: "reflex_decide",
    description:
      "Fast calibrated single-choice decision over a state (System 1: no text generation). Use for routing, triage, categorization.",
    inputSchema: {
      type: "object",
      required: ["state", "instructions", "options"],
      properties: {
        state: { type: "string", description: "The input text/JSON to decide over" },
        instructions: { type: "string", description: "The question to answer about the state" },
        options: { type: "object", patternProperties: { "^.*$": { type: "string", description: "option description" } }, description: "label → description mapping" },
      },
    },
    outputSchema: {
      type: "object",
      required: ["type", "choice", "probabilities", "confidence", "inputTokens"],
      properties: {
        type: { type: "string", const: "choice" },
        choice: { type: "string" },
        probabilities: { type: "object", patternProperties: { "^.*$": { type: "number" } } },
        confidence: { type: "number" },
        inputTokens: { type: "integer" },
      },
    },
  },
  {
    name: "reflex_judge",
    description: "Calibrated probability that a condition holds for a state (binary, no text generation).",
    inputSchema: {
      type: "object",
      required: ["state", "question"],
      properties: {
        state: { type: "string" },
        question: { type: "string", description: "Yes/no question about the state" },
      },
    },
    outputSchema: {
      type: "object",
      required: ["type", "probability", "confidence", "inputTokens"],
      properties: {
        type: { type: "string", const: "bool" },
        probability: { type: "number" },
        confidence: { type: "number" },
        inputTokens: { type: "integer" },
      },
    },
  },
  {
    name: "reflex_rate",
    description: "Calibrated rating on an ordinal rubric (expected level + distribution).",
    inputSchema: {
      type: "object",
      required: ["state", "instructions", "levels"],
      properties: {
        state: { type: "string" },
        instructions: { type: "string" },
        levels: { type: "array", items: { type: "string" }, minItems: 2, description: "ordered rubric levels, low → high" },
      },
    },
    outputSchema: {
      type: "object",
      required: ["type", "score", "levelCount", "probabilities", "confidence", "inputTokens"],
      properties: {
        type: { type: "string", const: "score" },
        score: { type: "number" },
        levelCount: { type: "integer" },
        probabilities: { type: "object", patternProperties: { "^.*$": { type: "number" } } },
        confidence: { type: "number" },
        inputTokens: { type: "integer" },
      },
    },
  },
  {
    name: "reflex_route",
    description: "Recommend a model tier (small/mid/frontier) + guardrail flags for an incoming message, in one fast local pass.",
    inputSchema: {
      type: "object",
      required: ["message"],
      properties: { message: { type: "string", description: "Incoming user message" } },
    },
    outputSchema: {
      type: "object",
      required: ["type", "tier", "reason", "guards", "inputTokens"],
      properties: {
        type: { type: "string", const: "route" },
        tier: { anyOf: [{ type: "string", const: "small" }, { type: "string", const: "mid" }, { type: "string", const: "frontier" }] },
        reason: { type: "string" },
        complexity: { type: "string" },
        needsCode: { type: "number" },
        longContext: { type: "number" },
        guards: {
          type: "object",
          required: ["injection", "harmful"],
          properties: { injection: { type: "number" }, harmful: { type: "number" } },
        },
        inputTokens: { type: "integer" },
      },
    },
  },
] as const;

interface JsonRpcRequest {
  jsonrpc?: string;
  id?: unknown;
  method: string;
  params?: unknown;
}

class McpError extends Error {
  constructor(readonly code: number, message: string) {
    super(message);
  }
}

/** Line-oriented JSON-RPC handler; testable without real stdio. */
export class ReflexMcpServer {
  constructor(private readonly cores: ToolCores) {}

  async handleLine(line: string): Promise<string | null> {
    let req: JsonRpcRequest;
    try {
      req = JSON.parse(line) as JsonRpcRequest;
    } catch {
      return JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } });
    }
    if (req.id === undefined || req.id === null) return null; // notification (incl. notifications/initialized)
    try {
      const result = await this.dispatch(req);
      return JSON.stringify({ jsonrpc: "2.0", id: req.id, result });
    } catch (e) {
      if (e instanceof McpError) return JSON.stringify({ jsonrpc: "2.0", id: req.id, error: { code: e.code, message: e.message } });
      return JSON.stringify({ jsonrpc: "2.0", id: req.id, error: { code: -32603, message: (e as Error).message } });
    }
  }

  private async dispatch(req: JsonRpcRequest): Promise<unknown> {
    switch (req.method) {
      case "initialize": {
        // MCP version negotiation (PR#2 review #5): echo the requested version only when
        // this server implements it; otherwise answer with the version we support so the
        // client can decide how to proceed.
        const p = (req.params ?? {}) as { protocolVersion?: string };
        const requested = typeof p.protocolVersion === "string" ? p.protocolVersion : undefined;
        const protocolVersion = requested && SUPPORTED_PROTOCOL_VERSIONS.has(requested) ? requested : PROTOCOL_VERSION;
        return { protocolVersion, capabilities: { tools: {} }, serverInfo: SERVER_INFO };
      }
      case "ping":
        return {};
      case "tools/list":
        return { tools: TOOL_SPECS };
      case "tools/call": {
        const p = (req.params ?? {}) as { name?: string; arguments?: Record<string, unknown> };
        const args = p.arguments ?? {};
        try {
          if (p.name === "reflex_decide") return this.toolResult(await this.cores.decide(args as never));
          if (p.name === "reflex_judge") return this.toolResult(await this.cores.judge(args as never));
          if (p.name === "reflex_rate") return this.toolResult(await this.cores.rate(args as never));
          if (p.name === "reflex_route") return this.toolResult(await this.cores.route(args as never));
          throw new McpError(-32602, `Unknown tool: ${String(p.name)}`);
        } catch (e) {
          if (e instanceof McpError) throw e;
          return {
            content: [{ type: "text", text: (e as Error).message }],
            isError: true,
          };
        }
      }
      default:
        throw new McpError(-32601, `Method not found: ${req.method}`);
    }
  }

  private toolResult(out: { text: string; data: Record<string, unknown> }) {
    return { content: [{ type: "text", text: out.text }], structuredContent: out.data };
  }
}

/** Resolve an engine the same way the pi extension does (env → local → HF download). */
export async function resolveEngineFromEnv(env: Record<string, string | undefined> = process.env): Promise<Engine> {
  const name = (env.PI_REFLEX_ENGINE as EngineName | undefined) ?? "multilingual";
  const quant: Quant = env.PI_REFLEX_QUANT === "fp32" ? "fp32" : "int8";
  const local = findLocalEngine(name, quant);
  if (local) return Engine.fromArtifacts(local, { int8: quant === "int8" });
  const dir = await ensureEngine(name, { quant });
  return Engine.fromArtifacts(dir, { int8: quant === "int8" });
}

/** Wire the server to stdio. */
export async function main(getEngine: () => Promise<Engine> = () => resolveEngineFromEnv()): Promise<void> {
  // One shared in-flight load (PR#2 review #6): concurrent cold tools/call requests must
  // not start parallel ensureEngine downloads into the same artifact paths. A failed load
  // resets the promise so a later request retries.
  let engineP: Promise<Engine> | null = null;
  const load = (): Promise<Engine> => {
    const p = getEngine().catch((e) => {
      engineP = null;
      throw e;
    });
    engineP = p;
    return p;
  };
  const sharedGetEngine = (): Promise<Engine> => engineP ?? load();

  const cores = createToolCores(sharedGetEngine);
  const server = new ReflexMcpServer(cores);
  const rl = createInterface({ input: process.stdin });
  rl.on("line", (line) => {
    void (async () => {
      const res = await server.handleLine(line);
      if (res !== null) process.stdout.write(res + "\n");
    })().catch((e) => process.stderr.write(`pi-reflex-mcp: ${String(e)}\n`));
  });
  await new Promise<void>((resolve) => rl.on("close", resolve));
}
