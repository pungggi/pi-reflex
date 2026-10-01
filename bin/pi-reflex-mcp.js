#!/usr/bin/env node
/**
 * pi-reflex MCP server entry — expose the System 1 decision tools over stdio
 * to any MCP client:  pi mcp add reflex -- node pi-reflex-mcp
 */
import { main } from "../dist/mcp/server.js";

main().catch((e) => {
  console.error("pi-reflex-mcp:", e);
  process.exit(1);
});
