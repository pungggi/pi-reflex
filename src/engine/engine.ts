/**
 * Engine — the pi-jev inference runtime. Mirrors laya's Agent.system_one exactly:
 * tokenize -> build sequences -> single fused forward pass -> temperature-calibrated answers.
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { buildAnswer } from "../core/answers.js";
import { QTYPES, type Temperatures } from "../core/calibration.js";
import { buildSequence, collate, renderOptions, type CollatedItem } from "../core/serialize.js";
import { toInternal, type Answer, type InternalQuestion, type QuestionDef, type Questions, type SystemOneResult } from "../core/types.js";
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
  private readonly tok: HfTokenizer;
  private readonly sess: OnnxSession;
  private disposed = false;

  private constructor(name: string, tok: HfTokenizer, sess: OnnxSession, cfg: EngineConfigJson) {
    this.name = name;
    this.tok = tok;
    this.sess = sess;
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

    const tok = HfTokenizer.fromDir(dir);
    const sess = await OnnxSession.create(model, { threads: opts.threads });
    const name = dir.split(/[\\/]/).filter(Boolean).pop() ?? "engine";
    return new Engine(name, tok, sess, cfg);
  }

  private assertReady(): void {
    if (this.disposed) throw new Error(`engine '${this.name}' has been disposed`);
  }

  private buildItem(state: unknown, q: InternalQuestion): CollatedItem {
    const { ids, markers } = buildSequence(this.tok, state, q, { maxLen: this.maxLen, headMaxLen: this.headMaxLen });
    if (markers.length !== renderOptions(q).length) {
      throw new Error(`question options exceed head_max_len=${this.headMaxLen}`);
    }
    return { ids, markers, qtype: QTYPES[q.t] };
  }

  /** Evaluate typed questions over state in a single forward pass (laya `system_one` parity). */
  async systemOne(state: unknown, questions: Questions): Promise<SystemOneResult> {
    this.assertReady();
    const ids = Object.keys(questions);
    if (ids.length === 0) throw new Error("questions must not be empty");

    const internals = ids.map((qid) => toInternal(questions[qid]!));
    const items = internals.map((q) => this.buildItem(state, q));

    const b = collate(items, this.tok.padTokenId);
    const { optionLogits, actLogits } = await this.sess.run(b);

    const answers: SystemOneResult["answers"] = {};
    ids.forEach((qid, r) => {
      const item = items[r]!;
      const q = internals[r]!;
      const optionRow = optionLogits[r];
      const actRow = actLogits[r];
      if (!optionRow || !actRow) throw new Error("engine: missing output row for question");
      answers[qid] = buildAnswer(q, optionRow.slice(0, item.markers.length), actRow, this.temps);
    });

    return {
      model: `pi-jev/${this.name}`,
      answers,
      usage: { input_tokens: b.totalTokens, output_tokens: 0 },
    };
  }

  /**
   * Contract D4 hot path: score M states against ONE question in a single forward
   * pass (batching M items costs ~1 forward pass, not M). `question` must be a
   * `score`, `noul` or `choice` definition; one answer per state, same order.
   */
  async batchQuestion(states: readonly unknown[], question: QuestionDef): Promise<Answer[]> {
    this.assertReady();
    if (states.length === 0) throw new Error("states must not be empty");
    const q = toInternal(question);
    const k = renderOptions(q).length;
    const items = states.map((s) => {
      const item = this.buildItem(s, q);
      if (item.markers.length !== k) throw new Error("inconsistent marker count across batched states");
      return item;
    });

    const b = collate(items, this.tok.padTokenId);
    const { optionLogits, actLogits } = await this.sess.run(b);

    return items.map((item, r) => {
      const optionRow = optionLogits[r];
      const actRow = actLogits[r];
      if (!optionRow || !actRow) throw new Error("engine: missing output row for batched state");
      return buildAnswer(q, optionRow.slice(0, item.markers.length), actRow, this.temps);
    });
  }

  /** Free native session resources. The engine is unusable afterwards. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.sess.dispose();
  }

  predict = this.systemOne;
}
