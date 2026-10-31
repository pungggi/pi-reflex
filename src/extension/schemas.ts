/**
 * The single source of truth for the reflex tools' parameter and output schemas.
 *
 * The pi extension registers these typebox objects verbatim; the standalone MCP
 * server carries hand-written JSON mirrors (it must stay free of pi/typebox
 * imports), and tests/mcp.test.ts asserts the two are deeply equal after a
 * JSON round-trip — so a schema change cannot drift between the two surfaces.
 */
import { Type } from "typebox";

export interface ToolSchema {
  parameters: ReturnType<typeof Type.Object>;
  outputSchema: ReturnType<typeof Type.Object>;
}

export const REFLEX_TOOL_SCHEMAS = {
  decide: {
    parameters: Type.Object({
      state: Type.String({ description: "The input text/JSON to decide over" }),
      instructions: Type.String({ description: "The question to answer about the state" }),
      options: Type.Record(Type.String(), Type.String({ description: "option description" }), { description: "label → description mapping" }),
    }),
    outputSchema: Type.Object({
      type: Type.Literal("choice"),
      choice: Type.String(),
      probabilities: Type.Record(Type.String(), Type.Number()),
      confidence: Type.Number(),
      inputTokens: Type.Integer(),
    }),
  },
  judge: {
    parameters: Type.Object({
      state: Type.String(),
      question: Type.String({ description: "Yes/no question about the state" }),
    }),
    outputSchema: Type.Object({
      type: Type.Literal("bool"),
      probability: Type.Number(),
      confidence: Type.Number(),
      inputTokens: Type.Integer(),
    }),
  },
  rate: {
    parameters: Type.Object({
      state: Type.String(),
      instructions: Type.String(),
      levels: Type.Array(Type.String(), { minItems: 2, description: "ordered rubric levels, low → high" }),
    }),
    outputSchema: Type.Object({
      type: Type.Literal("score"),
      score: Type.Number(),
      levelCount: Type.Integer(),
      probabilities: Type.Record(Type.String(), Type.Number()),
      confidence: Type.Number(),
      inputTokens: Type.Integer(),
    }),
  },
  route: {
    parameters: Type.Object({
      message: Type.String({ description: "Incoming user message" }),
    }),
    outputSchema: Type.Object({
      type: Type.Literal("route"),
      tier: Type.Union([Type.Literal("small"), Type.Literal("mid"), Type.Literal("frontier")]),
      reason: Type.String(),
      complexity: Type.Optional(Type.String()),
      needsCode: Type.Optional(Type.Number()),
      longContext: Type.Optional(Type.Number()),
      guards: Type.Object({ injection: Type.Number(), harmful: Type.Number() }),
      inputTokens: Type.Integer(),
    }),
  },
} satisfies Record<string, ToolSchema>;

export type ReflexToolName = keyof typeof REFLEX_TOOL_SCHEMAS;
