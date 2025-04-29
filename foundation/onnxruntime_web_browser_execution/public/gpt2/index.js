import { AutoTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';



async function createSession(modelPath, preferredOutputLocation, executionProviders = "webnn") {
    return await ort.InferenceSession.create(modelPath, {
        executionProviders: [executionProviders],
        preferredOutputLocation: preferredOutputLocation
    });
}

function argMax(array) {
    return [].reduce.call(array, (m, c, i, arr) => c > arr[m] ? i : m, 0)
}
async function runModel(text) {
    // TODO: figure out how to use webnn (not working)
    // TODO: figure out how to use webgpu (really slow)
    // TODO: use kv cache?
    const session = await createSession('model/gpt2/gpt2.onnx', 'cpu', 'wasm');

    const tokenizer = await AutoTokenizer.from_pretrained('gpt2');
    let encoded = await tokenizer(text).input_ids;
    for (let i = 0; i < 20; i++) {
        const outputs = await session.run({ input_ids: encoded })

        const logits = outputs.logits.cpuData;
        const vocabSize = outputs.logits.dims[2];
        const logits_next_token = logits.slice(-vocabSize)
        const nextToken = argMax(logits_next_token);
        console.log("Generated token IDs:", nextToken);
        console.log(tokenizer.decode([nextToken]))

        const data = BigInt64Array.from([...encoded.data, BigInt(nextToken)])
        encoded = new ort.Tensor('int64', data, [1, data.length]);
    }
    return tokenizer.decode([...encoded.cpuData])
}
async function startModel() {
    const prompt = document.getElementById("prompt").value;
    const text = await runModel(prompt);

    document.getElementById("pre").innerHTML = text;
}
window.startModel = startModel



function arrayEquals(a, b) {
    return a.length === b.length && a.every((val, index) => val === b[index]);
}

async function testModel() {
    const text = "Why is the sky blue?";
    const tokenizer = await AutoTokenizer.from_pretrained('gpt2');
    const encoded = await tokenizer(text).input_ids;
    const feeds = { input_ids: encoded };

    // Test full model
    const session = await createSession('model/gpt2/gpt2.onnx', 'cpu', 'wasm');
    const out = await session.run(feeds);
    console.log(out);


    // Test split model
    const session_p1 = await createSession('model/gpt2/gpt2_p1.onnx', 'cpu', 'wasm');
    const out_p1 = await session_p1.run(feeds);
    console.log(out_p1);

    const session_p2 = await createSession('model/gpt2/gpt2_p2.onnx', 'cpu', 'wasm');
    const out_p2 = await session_p2.run(out_p1);
    console.log(out_p2);

    // assert
    console.assert(arrayEquals(out.logits.cpuData, out_p2.logits.cpuData));
    console.log("Split gpt2 gives same result as full variant")
}
window.testModel = testModel
