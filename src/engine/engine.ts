/**
 * Engine — the pi-jev inference runtime. Mirrors laya's Agent.system_one exactly:
 * tokenize -> build sequences -> single fused forward pass -> temperature-calibrated answers.
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { buildAnswer } from "../core/answers.js";
import { QTYPES, type Temperatures } from "../core/calibration.js";
import { buildSequence, collate, renderOptions } from "../core/serialize.js";
import { toInternal, type Questions, type SystemOneResult } from "../core/types.js";
import { HfTokenizer } from "./tokenizer.js";
import { OnnxSession } from "./session.js";

export interface EngineConfigJson {
  temperature?: [number, number, number];
  temperature_by_options?: Record<string, number>;
  max_len?: number;
  head_max_len?: number;
}

export interface EngineOptions {
  /** use the int8-quantized graph (model.int8.onnx) instead of fp32 */
  int8?: boolean;
  threads?: number;
}

export class Engine {
  readonly name: string;
  readonly maxLen: number;
  readonly headMaxLen: number;
  private readonly temps: Temperatures;

  private constructor(
    name: string,
    private readonly tok: HfTokenizer,
    private readonly sess: OnnxSession,
    cfg: EngineConfigJson,
  ) {
    this.name = name;
    this.maxLen = cfg.max_len ?? 512;
    this.headMaxLen = cfg.head_max_len ?? 192;
    this.temps = {
      temperature: cfg.temperature ?? [1, 1, 1],
      temperatureByOptions: cfg.temperature_by_options ?? {},
    };
  }

  /** Load from an artifacts dir: model.onnx(+.data), rl_agent_config.json, tokenizer/. */
  static async fromArtifacts(dir: string, opts: EngineOptions = {}): Promise<Engine> {
    const cfgPath = join(dir, "rl_agent_config.json");
    if (!existsSync(cfgPath)) throw new Error(`missing rl_agent_config.json in ${dir}`);
    const cfg = JSON.parse(readFileSync(cfgPath, "utf8")) as EngineConfigJson;

    const model = opts.int8 ? join(dir, "model.int8.onnx") : join(dir, "model.onnx");
    if (!existsSync(model)) throw new Error(`missing ${model}`);

    const [tok, sess] = await Promise.all([
      Promise.resolve(HfTokenizer.fromDir(dir)),
      OnnxSession.create(model, { threads: opts.threads }),
    ]);
    const name = dir.split(/[\\/]/).filter(Boolean).pop() ?? "engine";
    return new Engine(name, tok, sess, cfg);
  }

  /** Evaluate typed questions over state in a single forward pass (laya `system_one` parity). */
  async systemOne(state: unknown, questions: Questions): Promise<SystemOneResult> {
    const ids = Object.keys(questions);
    if (ids.length === 0) throw new Error("questions must not be empty");

    const internals = ids.map((qid) => toInternal(questions[qid]));
    const items = internals.map((q) => {
      const { ids: seq, markers } = buildSequence(this.tok, state, q, { maxLen: this.maxLen, headMaxLen: this.headMaxLen });
      if (markers.length !== renderOptions(q).length) {
        throw new Error(`question options exceed head_max_len=${this.headMaxLen}`);
      }
      return { ids: seq, markers, qtype: QTYPES[q.t] };
    });

    const b = collate(items, this.tok.padTokenId);
    const { optionLogits, actLogits } = await this.sess.run(b);

    const answers: SystemOneResult["answers"] = {};
    ids.forEach((qid, r) => {
      const k = items[r].markers.length;
      answers[qid] = buildAnswer(internals[r], optionLogits[r].slice(0, k), actLogits[r], this.temps);
    });

    return {
      model: `pi-jev/${this.name}`,
      answers,
      usage: { input_tokens: b.totalTokens, output_tokens: 0 },
    };
  }

  predict = this.systemOne;
}
