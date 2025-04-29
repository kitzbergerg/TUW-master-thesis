import { AutoTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';
import file from "./model_external_data.json" with { type: "json" };


async function createSession(modelPath, preferredOutputLocation, executionProviders = "webnn") {
  return await ort.InferenceSession.create(modelPath, {
    executionProviders: [executionProviders],
    preferredOutputLocation: preferredOutputLocation,
    externalData: file.full
  });
}

async function startModel() {
  throw Error("cannot run model, model to large (>4GB)")
}
window.startModel = startModel



async function testModel() {
  const text = "Why is the sky blue?";
  const tokenizer = await AutoTokenizer.from_pretrained('google/gemma-3-1b-it');
  const encoded = await tokenizer(text).input_ids;
  const feeds = { input_ids: encoded };

  // Test full model
  console.log("Unable to test full model. To large (>4GB)");


  // Test split model
  const session_p1 = await createSession('model/gemma3/p1/gemma-3-1b-it-p1.onnx', 'cpu', 'wasm');
  const out_p1 = await session_p1.run(feeds);
  console.log(out_p1);

  const session_p2 = await createSession('model/gemma3/p1/gemma-3-1b-it-p2.onnx', 'cpu', 'wasm');
  const out_p2 = await session_p2.run(out_p1);
  console.log(out_p2);
}
window.testModel = testModel
