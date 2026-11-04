// Runs the ONNX detector off the main thread so the page animations keep going during a check.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.mjs";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
ort.env.wasm.numThreads = (self.crossOriginIsolated && navigator.hardwareConcurrency) ? Math.min(4, navigator.hardwareConcurrency) : 1;
let session = null;
self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === "init") {
      session = await ort.InferenceSession.create(e.data.buffer, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      self.postMessage({ id, ok: true });
    } else if (type === "run") {
      const { views, n } = e.data;
      const out = await session.run({ pixel_values: new ort.Tensor("float32", views, [n, 3, 224, 224]) });
      self.postMessage({ id, ok: true, logit: Array.from(out.logit.data), patch: Array.from(out.patch.data) });
    }
  } catch (err) { self.postMessage({ id, ok: false, error: String(err && err.message || err) }); }
};
