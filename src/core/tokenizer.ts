/**
 * Minimal tokenizer contract needed by the serializer.
 * Production implementation wraps `@huggingface/tokenizers` (Rust bindings, no Python).
 */
export interface TokenizerLike {
  readonly clsTokenId: number;
  readonly sepTokenId: number;
  readonly maskTokenId: number;
  readonly padTokenId: number;
  /** the literal mask token string, e.g. "[MASK]" (used to scrub it out of user text) */
  readonly maskToken: string;
  /** tokenize WITHOUT special tokens */
  encode(text: string): number[];
}
