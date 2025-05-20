import * as ort from 'onnxruntime-web';

const numLayers = 32;
const hiddenSize = 2560;
const numHeads = 32;
const headDim = hiddenSize / numHeads;

type SessionInput<T> = {
    readonly [name: string]: T;
}
interface SessionInputEntry {
    data: number[],
    dims: number[]
}

type ModelInput = {
    [name: string]: ort.Tensor;
}

type SessionOutput = {
    [name: string]: SessionEntry;
}
interface SessionEntry {
    data: number[],
    dims: readonly number[]
}

export class InferenceSession {
    session: ort.InferenceSession;
    pastKeyValues: Map<string, Map<string, ort.Tensor>>;

    constructor(session: ort.InferenceSession, pastKeyValues: Map<string, Map<string, ort.Tensor>>) {
        this.session = session;
        this.pastKeyValues = pastKeyValues;
    }

    static async createSession(modelPath: string, externalData: ort.ExternalDataFileType[]): Promise<InferenceSession> {
        const session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['webgpu'],
            preferredOutputLocation: 'gpu-buffer',
            externalData
        });
        return new InferenceSession(session, new Map());
    }


    async runInference(requestId: string, input: SessionInput<number[] | SessionInputEntry>): Promise<SessionOutput> {
        console.log("Inference request: ", requestId, input)

        const isFirstBlock = 'input_ids' in input;
        const isFirstRequest = !this.pastKeyValues.has(requestId)

        let feeds: ModelInput = {};
        if (isFirstBlock) {
            const input_cast = input as SessionInput<number[]>;
            const seqLen = input_cast['input_ids'].length;
            for (const [key, value] of Object.entries(input_cast)) {
                feeds[key] = new ort.Tensor('int64', BigInt64Array.from(value.map(BigInt)), [1, seqLen])
            }
        } else {
            const input_cast = input as SessionInput<SessionInputEntry>;
            for (const [key, value] of Object.entries(input_cast)) {
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
            const cache = this.pastKeyValues.get(requestId);
            for (let layer = layerStart; layer < layerEnd; layer++) {
                feeds[`past_key_values.${layer}.key`] = cache.get(`present.${layer}.key`);
                feeds[`past_key_values.${layer}.value`] = cache.get(`present.${layer}.value`);
            }
        }
        console.log("Model input: ", feeds)
        const inferenceResults = await this.session.run(feeds);
        console.log("Model output: ", inferenceResults)

        const result: SessionOutput = {}
        if (isFirstRequest) this.pastKeyValues.set(requestId, new Map())
        const cache = this.pastKeyValues.get(requestId);
        for (const name in inferenceResults) {
            if (name.startsWith('present.')) {
                cache.set(name, inferenceResults[name]);
            } else {
                result[name] = {
                    data: Array.from(await inferenceResults[name].getData() as Float32Array),
                    dims: inferenceResults[name].dims
                }
            }
        }

        console.log("Inference response: ", result)
        return result

    }
}