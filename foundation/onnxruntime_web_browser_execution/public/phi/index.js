import { AutoTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';
import externalData from './data.json'
import externalDataP1 from './data_p1.json'
import externalDataP2 from './data_p2.json'


const numLayers = 32;
const vocabSize = 51200;
const hiddenSize = 2560;
const numHeads = 32;
const headDim = hiddenSize / numHeads;
const one = BigInt(1);
const tokenizer = await AutoTokenizer.from_pretrained('microsoft/phi-2');



async function createSession(modelPath, externalData) {
    return await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgpu'],
        preferredOutputLocation: 'gpu-buffer',
        externalData
    });
}

function argMax(array) {
    return [].reduce.call(array, (m, c, i, arr) => c > arr[m] ? i : m, 0)
}

async function runModelSplit(prompt, maxTokens = 100) {
    const s1 = await createSession('http://localhost:3000/phi/split/p1/model.onnx', externalDataP1);
    const s2 = await createSession('http://localhost:3000/phi/split/p2/model.onnx', externalDataP2);
    let inputSequence = [...(await tokenizer(prompt)).input_ids.data];

    const pastKeyValues = {};
    for (let i = 0; i < maxTokens; i++) {
        const isFirst = i === 0;
        const inputIds = isFirst ? inputSequence : [inputSequence[inputSequence.length - 1]];

        const seqLen = inputIds.length;
        const currentInput = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, seqLen]);
        const attentionMask = new ort.Tensor('int64', BigInt64Array.from(new Array(seqLen).fill(one)), [1, seqLen]);
        const positionIds = new ort.Tensor('int64', BigInt64Array.from(isFirst ? [...Array(seqLen).keys()].map(x => BigInt(x)) : [BigInt(inputSequence.length - 1)]), [1, seqLen]);

        const input = { input_ids: currentInput, attention_mask: attentionMask, position_ids: positionIds };
        for (let layer = 0; layer < numLayers / 2; layer++) {
            const keyName = `past_key_values.${layer}.key`;
            const valueName = `past_key_values.${layer}.value`;
            input[keyName] = isFirst ? new ort.Tensor('float32', new Float32Array(0), [1, numHeads, 0, headDim]) : pastKeyValues[`present.${layer}.key`];
            input[valueName] = isFirst ? new ort.Tensor('float32', new Float32Array(0), [1, numHeads, 0, headDim]) : pastKeyValues[`present.${layer}.value`];
        }
        const intermediate = await s1.run(input);

        const passOn = {}
        for (const name in intermediate) {
            if (name.startsWith('present.')) pastKeyValues[name] = intermediate[name];
            else passOn[name] = intermediate[name];
        }
        for (let layer = numLayers / 2; layer < numLayers; layer++) {
            const keyName = `past_key_values.${layer}.key`;
            const valueName = `past_key_values.${layer}.value`;
            passOn[keyName] = isFirst ? new ort.Tensor('float32', new Float32Array(0), [1, numHeads, 0, headDim]) : pastKeyValues[`present.${layer}.key`];
            passOn[valueName] = isFirst ? new ort.Tensor('float32', new Float32Array(0), [1, numHeads, 0, headDim]) : pastKeyValues[`present.${layer}.value`];
        }
        const output = await s2.run(passOn);
        for (const name in output) {
            if (name.startsWith('present.')) pastKeyValues[name] = output[name];
        }

        const logits = await output.logits.getData();
        const lastLogits = isFirst ? logits.slice(-vocabSize) : logits;
        const nextToken = argMax(lastLogits);
        console.log(`Generated token: ${tokenizer.decode([nextToken])}`);
        inputSequence.push(nextToken);
    }

    return tokenizer.decode(inputSequence);
}

async function runModel(prompt, maxTokens = 100) {
    const session = await createSession('http://localhost:3000/phi/torch/model.onnx', externalData);
    let inputSequence = [...(await tokenizer(prompt)).input_ids.data];

    const pastKeyValues = {};
    for (let i = 0; i < maxTokens; i++) {
        const isFirst = i === 0;
        const inputIds = isFirst ? inputSequence : [inputSequence[inputSequence.length - 1]];

        const seqLen = inputIds.length;
        const currentInput = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, seqLen]);
        const attentionMask = new ort.Tensor('int64', BigInt64Array.from(new Array(seqLen).fill(one)), [1, seqLen]);
        const positionIds = new ort.Tensor('int64', BigInt64Array.from(isFirst ? [...Array(seqLen).keys()].map(x => BigInt(x)) : [BigInt(inputSequence.length - 1)]), [1, seqLen]);

        const feeds = { input_ids: currentInput, attention_mask: attentionMask, position_ids: positionIds };

        for (let layer = 0; layer < numLayers; layer++) {
            const keyName = `past_key_values.${layer}.key`;
            const valueName = `past_key_values.${layer}.value`;
            feeds[keyName] = isFirst ? new ort.Tensor('float32', new Float32Array(0), [1, numHeads, 0, headDim]) : pastKeyValues[`present.${layer}.key`];
            feeds[valueName] = isFirst ? new ort.Tensor('float32', new Float32Array(0), [1, numHeads, 0, headDim]) : pastKeyValues[`present.${layer}.value`];
        }

        const outputs = await session.run(feeds);
        for (const name in outputs) {
            if (name.startsWith('present.')) pastKeyValues[name] = outputs[name];
        }

        const logits = await outputs.logits.getData();
        const lastLogits = isFirst ? logits.slice(-vocabSize) : logits;
        const nextToken = argMax(lastLogits);
        console.log(`Generated token: ${tokenizer.decode([nextToken])}`);
        inputSequence.push(nextToken);
    }

    return tokenizer.decode(inputSequence);
}

async function startModel() {
    const prompt = document.getElementById("prompt").value;
    const text = await runModelSplit(prompt);

    document.getElementById("pre").innerHTML = text;
}
window.startModel = startModel



function arrayEquals(a, b) {
    return a.length === b.length && a.every((val, index) => val === b[index]);
}

async function testModel() {
    // TODO
}
window.testModel = testModel
