/**
 * ONNX Runtime session wrapper — flattens our collated batch into ORT tensors and
 * unflattens the fused DecisionModel graph outputs.
 */
import * as ort from "onnxruntime-node";
import type { CollatedBatch } from "../core/serialize.js";

export interface SessionRun {
  /** [batch][kmax] option logits */
  optionLogits: number[][];
  /** [batch][n_act] action logits */
  actLogits: number[][];
}

function to2D(data: Float32Array, dims: readonly number[]): number[][] {
  if (dims.length !== 2) throw new Error(`expected a 2-D output tensor, got dims [${dims}]`);
  const rows = dims[0]!;
  const cols = data.length / rows;
  const out: number[][] = [];
  for (let r = 0; r < rows; r++) out.push(Array.from(data.subarray(r * cols, (r + 1) * cols)));
  return out;
}

export class OnnxSession {
  private constructor(private readonly sess: ort.InferenceSession) {}

  static async create(modelPath: string, opts: { threads?: number } = {}): Promise<OnnxSession> {
    const sessionOptions: ort.InferenceSession.SessionOptions = {};
    if (opts.threads) sessionOptions.intraOpNumThreads = opts.threads;
    const sess = await ort.InferenceSession.create(modelPath, sessionOptions);
    return new OnnxSession(sess);
  }

  async run(b: CollatedBatch): Promise<SessionRun> {
    const B = b.batch;
    const L = b.seqLen;
    const K = b.kmax;

    const inputIds = new BigInt64Array(B * L);
    const att = new BigInt64Array(B * L);
    for (let i = 0; i < B; i++) {
      const idRow = b.inputIds[i]!;
      const attRow = b.attentionMask[i]!;
      for (let j = 0; j < L; j++) {
        inputIds[i * L + j] = BigInt(idRow[j] ?? 0);
        att[i * L + j] = BigInt(attRow[j] ?? 0);
      }
    }
    const markerPos = new BigInt64Array(B * K);
    const markerMask = new Uint8Array(B * K);
    for (let i = 0; i < B; i++) {
      const posRow = b.markerPos[i]!;
      const maskRow = b.markerMask[i]!;
      for (let j = 0; j < K; j++) {
        markerPos[i * K + j] = BigInt(posRow[j] ?? 0);
        markerMask[i * K + j] = (maskRow[j] ?? false) ? 1 : 0;
      }
    }
    const qtype = new BigInt64Array(B);
    for (let i = 0; i < B; i++) qtype[i] = BigInt(b.qtype[i] ?? 0);

    const out = await this.sess.run({
      input_ids: new ort.Tensor("int64", inputIds, [B, L]),
      attention_mask: new ort.Tensor("int64", att, [B, L]),
      marker_pos: new ort.Tensor("int64", markerPos, [B, K]),
      marker_mask: new ort.Tensor("bool", markerMask, [B, K]),
      qtype: new ort.Tensor("int64", qtype, [B]),
    });

    const logitsT = out.option_logits;
    const actT = out.act_logits;
    if (!logitsT || !actT) {
      throw new Error(
        `ONNX graph outputs mismatch: expected 'option_logits' and 'act_logits', got [${this.sess.outputNames.join(", ")}]`,
      );
    }

    return {
      optionLogits: to2D(logitsT.data as Float32Array, logitsT.dims),
      actLogits: to2D(actT.data as Float32Array, actT.dims),
    };
  }

  /** Release native session resources (Router LRU eviction; safe to call twice). */
  dispose(): void {
    try {
      (this.sess as unknown as { release?: () => void }).release?.();
    } catch {
      // older onnxruntime builds without release() — GC handles it
    }
  }
}
