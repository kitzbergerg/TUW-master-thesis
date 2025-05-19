import * as ort from 'onnxruntime-web';

const numLayers = 32;
const hiddenSize = 2560;
const numHeads = 32;
const headDim = hiddenSize / numHeads;

async function createSession(modelPath, externalData) {
    return await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgpu'],
        preferredOutputLocation: 'gpu-buffer',
        externalData
    });
}

const pastKeyValues = new Map()

async function runInference(session, requestId, input) {
    console.log("Inference request: ", requestId, input)

    const isFirstBlock = 'input_ids' in input;
    const isFirstRequest = !pastKeyValues.has(requestId)

    let feeds;
    if (isFirstBlock) {
        const seqLen = input.input_ids.length;
        feeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from(input.input_ids), [1, seqLen]),
            attention_mask: new ort.Tensor('int64', BigInt64Array.from(input.attention_mask), [1, seqLen]),
            position_ids: new ort.Tensor('int64', BigInt64Array.from(input.position_ids), [1, seqLen]),
        }
    } else {
        feeds = {};
        for (const [key, value] of Object.entries(input)) {
            feeds[key] = new ort.Tensor('float32', Float32Array.from(value.data), value.dims)
        }
    }

    // TODO: figure out how to get num of layers
    const layerStart = isFirstBlock ? 0 : numLayers / 2;
    const layerEnd = isFirstBlock ? numLayers / 2 : numLayers;
    if (isFirstRequest) {
        for (let layer = layerStart; layer < layerEnd; layer++) {
            feeds[`past_key_values.${layer}.key`] = new ort.Tensor('float32', new Float32Array(0), [1, numHeads, 0, headDim]);
            feeds[`past_key_values.${layer}.value`] = new ort.Tensor('float32', new Float32Array(0), [1, numHeads, 0, headDim]);
        }
    } else {
        const cache = pastKeyValues.get(requestId);
        for (let layer = layerStart; layer < layerEnd; layer++) {
            feeds[`past_key_values.${layer}.key`] = cache[`present.${layer}.key`];
            feeds[`past_key_values.${layer}.value`] = cache[`present.${layer}.value`];
        }
    }
    console.log("Model input: ", feeds)
    const inferenceResults = await session.run(feeds);
    console.log("Model output: ", inferenceResults)

    const result = {}
    if (isFirstRequest) pastKeyValues.set(requestId, {})
    const cache = pastKeyValues.get(requestId);
    for (const name in inferenceResults) {
        if (name.startsWith('present.')) {
            cache[name] = inferenceResults[name];
        } else {
            result[name] = {
                data: Array.from(await inferenceResults[name].getData()),
                dims: inferenceResults[name].dims
            }
        }
    }

    console.log("Inference response: ", result)
    return result

}

window.createSession = createSession
window.runInference = runInference
