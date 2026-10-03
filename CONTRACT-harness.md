# The pi-continual-harness consumer contract

[pi-continual-harness](https://github.com/pungggi/pi-continual-harness) is the first integration target for `pi-reflex`, and serves as a real-world blueprint for how to integrate `pi-reflex`'s capabilities into your own packages.

This contract defines how a `pi-reflex` companion package should integrate with the harness:

- **Registration**: The companion package registers itself as the deduplication `similarity` provider (and/or as a `jev` proposer via `registerProposer`).
- **Drop-in Upgrade**: It must act as a drop-in upgrade over the default token Jaccard similarity. Plain numeric returns must keep working to maintain compatibility.
- **Soft-fail Composition**: Following the `pi-mem` pattern, the harness gracefully falls back to token Jaccard whenever `pi-reflex` is absent, offline, or over budget.

### Why this matters to you

For end-users and developers integrating these tools, this contract provides a critical reliability guarantee: **you can adopt AI-powered features without introducing a single point of failure.**

By enforcing a strict drop-in upgrade with a transparent fallback, you get the benefits of semantic, calibrated decisions when the model is healthy. If the engine ever crashes, takes too long, or runs out of budget, your application won't break—it simply degrades gracefully back to the fast, reliable token Jaccard algorithm.

> **What is Token Jaccard?**  
> Token Jaccard is a simple, highly reliable algorithm that measures similarity by checking how many exact words two blocks of text share. It only understands structure, not meaning (e.g., "It is raining" and "Water falls from the sky" would score 0% similarity). `pi-reflex` provides the AI intelligence to understand true meaning, while Token Jaccard acts as the "dumb but unbreakable" safety net.
