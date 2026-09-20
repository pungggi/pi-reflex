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
  const rows = dims.length === 2 ? dims[0] : 1;
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
      for (let j = 0; j < L; j++) {
        inputIds[i * L + j] = BigInt(b.inputIds[i][j]);
        att[i * L + j] = BigInt(b.attentionMask[i][j]);
      }
    }
    const markerPos = new BigInt64Array(B * K);
    const markerMask = new Uint8Array(B * K);
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < K; j++) {
        markerPos[i * K + j] = BigInt(b.markerPos[i][j]);
        markerMask[i * K + j] = b.markerMask[i][j] ? 1 : 0;
      }
    }
    const qtype = new BigInt64Array(B);
    for (let i = 0; i < B; i++) qtype[i] = BigInt(b.qtype[i]);

    const out = await this.sess.run({
      input_ids: new ort.Tensor("int64", inputIds, [B, L]),
      attention_mask: new ort.Tensor("int64", att, [B, L]),
      marker_pos: new ort.Tensor("int64", markerPos, [B, K]),
      marker_mask: new ort.Tensor("bool", markerMask, [B, K]),
      qtype: new ort.Tensor("int64", qtype, [B]),
    });

    return {
      optionLogits: to2D(out.option_logits.data as Float32Array, out.option_logits.dims),
      actLogits: to2D(out.act_logits.data as Float32Array, out.act_logits.dims),
    };
  }
}
