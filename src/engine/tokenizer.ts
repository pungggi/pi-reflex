/**
 * Tokenizer adapter — wraps @huggingface/tokenizers (pure JS, the transformers.js v3
 * engine) to expose the minimal TokenizerLike contract the serializer needs.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";
import type { TokenizerLike } from "../core/tokenizer.js";

interface AddedTokenJson {
  id: number;
  content: string;
}

export class HfTokenizer implements TokenizerLike {
  readonly clsTokenId: number;
  readonly sepTokenId: number;
  readonly maskTokenId: number;
  readonly padTokenId: number;
  readonly maskToken: string;
  private readonly tok: Tokenizer;

  private constructor(tok: Tokenizer, specials: { cls: number; sep: number; mask: number; pad: number; maskStr: string }) {
    this.tok = tok;
    this.clsTokenId = specials.cls;
    this.sepTokenId = specials.sep;
    this.maskTokenId = specials.mask;
    this.padTokenId = specials.pad;
    this.maskToken = specials.maskStr;
  }

  /** Load from an artifacts directory containing tokenizer/tokenizer.json + tokenizer_config.json. */
  static fromDir(artifactsDir: string): HfTokenizer {
    const dir = join(artifactsDir, "tokenizer");
    const json = JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8")) as {
      added_tokens: AddedTokenJson[];
      model?: { vocab?: Record<string, number> };
    };
    const cfg = JSON.parse(readFileSync(join(dir, "tokenizer_config.json"), "utf8")) as Record<string, unknown>;

    const byContent = new Map(json.added_tokens.map((t) => [t.content, t.id]));
    const modelVocab = json.model?.vocab;
    const need = (token: unknown, fallback: string): string => {
      if (typeof token === "string") return token;
      if (token && typeof token === "object" && "content" in token) return String((token as { content: unknown }).content);
      return fallback;
    };
    const clsStr = need(cfg.cls_token, "[CLS]");
    const sepStr = need(cfg.sep_token, "[SEP]");
    const maskStr = need(cfg.mask_token, "[MASK]");
    const padStr = need(cfg.pad_token, "[PAD]");

    const idOf = (s: string): number => {
      // M2: prefer added_tokens, fall back to the model vocab (some tokenizers keep
      // specials in the model rather than as added tokens).
      const id = byContent.get(s) ?? modelVocab?.[s];
      if (id === undefined) throw new Error(`special token '${s}' not found in tokenizer added_tokens or model vocab`);
      return id;
    };

    const tok = new Tokenizer(json as unknown as Record<string, unknown>, cfg);
    return new HfTokenizer(tok, { cls: idOf(clsStr), sep: idOf(sepStr), mask: idOf(maskStr), pad: idOf(padStr), maskStr });
  }

  /** Tokenize WITHOUT special tokens (laya parity: `tok(text, add_special_tokens=False)`). */
  encode(text: string): number[] {
    return this.tok.encode(text, { add_special_tokens: false }).ids.map(Number);
  }
}
