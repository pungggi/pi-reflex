// Dev-only smoke: exercise the built extension against a stubbed ExtensionAPI.
// Verifies registration wiring (provider, virtual model, tools, guard, MCP) without pi.
import activate from "../dist/extension/index.js";

const registered = { providers: [], vmodels: [], tools: [], commands: [], handlers: [], mcp: [], toolRenderers: [] };
const pi = {
  registerProvider: (name, config) => registered.providers.push({ name, models: config.models?.length, classifiers: Object.keys(config.classifiers ?? {}) }),
  registerVirtualModel: (def) => registered.vmodels.push(`${def.provider}/${def.id}`),
  registerTool: (def) => registered.tools.push({ name: def.name, exposure: def.exposure, hasSchema: !!def.outputSchema, ns: def.namespace?.name }),
  registerCommand: (name) => registered.commands.push(name),
  registerMcpServer: (name, cfg) => registered.mcp.push({ name, cfg }),
  registerToolRenderer: (resolver) => registered.toolRenderers.push(resolver),
  on: (event) => registered.handlers.push(event),
};

activate(pi, { env: { PI_REFLEX_GUARD: "1", PI_REFLEX_MCP: "1", PI_REFLEX_TIER_SMALL: "anthropic/claude-haiku-4-5" } });
console.log(JSON.stringify(registered, null, 2));
const want = [
  registered.providers.length === 1 && registered.providers[0].models === 3,
  registered.vmodels.includes("reflex/auto"),
  registered.tools.length === 4 && registered.tools.every((t) => t.hasSchema && t.ns === "reflex"),
  registered.commands.includes("reflex"),
  registered.handlers.includes("session_start") && registered.handlers.includes("context_with_system"),
  registered.mcp.length === 1 && registered.mcp[0].name === "reflex",
  // pi ≥ 1.0.1: one renderer resolver covering bare + mcp__<server>__ reflex names
  registered.toolRenderers.length === 1 &&
    registered.toolRenderers.every((r) => r("reflex_judge", () => undefined) && r("mcp__reflex__reflex_judge", () => undefined) && r("read", () => undefined) === undefined),
];
if (!want.every(Boolean)) {
  console.error("SMOKE EXT FAIL", want);
  process.exit(1);
}
console.log("SMOKE EXT OK");
