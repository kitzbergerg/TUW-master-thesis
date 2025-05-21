import * as ort from 'onnxruntime-web';
import { ComputationMessage, ComputationMessageSchema, FirstModelInput, IntermediateModelData, IntermediateModelDataSchema, IntermediateResultSchema } from './gen/data_pb';
import { create } from '@bufbuild/protobuf';

const numLayers = 32;
const hiddenSize = 2560;
const numHeads = 32;
const headDim = hiddenSize / numHeads;

type OnnxSessionInput = { [name: string]: ort.OnnxValue };

export class InferenceSession {
    session: ort.InferenceSession;
    pastKeyValues: Map<string, ort.InferenceSession.OnnxValueMapType>;

    constructor(session: ort.InferenceSession, pastKeyValues: Map<string, ort.InferenceSession.OnnxValueMapType>) {
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


    async runInference(message: ComputationMessage): Promise<ComputationMessage> {
        const requestId = message.requestId;
        const input = message.data;

        console.log("Inference request: ", requestId, input)

        const isFirstRequest = !this.pastKeyValues.has(requestId)

        let feeds: OnnxSessionInput = {};
        switch (input.case) {
            case "first": {
                const inputIds = isFirstRequest ? input.value.inputIds : input.value.inputIds.slice(-1);
                const seqLen = inputIds.length;
                const attentionMask = new Array(seqLen).fill(1);
                const positionIds = Array.from(isFirstRequest ? Array(seqLen).keys() : [input.value.inputIds.length - 1]);

                feeds['input_ids'] = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, seqLen])
                feeds['attention_mask'] = new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, seqLen])
                feeds['position_ids'] = new ort.Tensor('int64', BigInt64Array.from(positionIds.map(BigInt)), [1, seqLen])
                break;
            }
            case "intermediate": {
                for (const [key, value] of Object.entries(input.value.map)) {
                    feeds[key] = new ort.Tensor('float32', Float32Array.from(value.data), value.dims)
                }
                break;
            }
        }

        // TODO: get layer/split information from server
        const isFirstBlock = input.case == "first";
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
                feeds[`past_key_values.${layer}.key`] = cache[`present.${layer}.key`];
                feeds[`past_key_values.${layer}.value`] = cache[`present.${layer}.value`];
            }
        }
        console.log("Model input: ", feeds)
        const inferenceResults = await this.session.run(feeds);
        console.log("Model output: ", inferenceResults)

        const split = Object.groupBy(Object.entries(inferenceResults), (el) => el[0].startsWith('present.') ? "cache" : "output")
        const newCache = Object.fromEntries(split.cache);
        const output = Object.fromEntries(split.output);

        this.pastKeyValues.set(requestId, newCache);


        const value = create(IntermediateModelDataSchema, {
            map: {}
        });
        for (const name in output) {
            value.map[name] = create(IntermediateResultSchema, {
                data: Array.from(await inferenceResults[name].getData() as Float32Array),
                dims: inferenceResults[name].dims as number[]
            })
        }

        console.log("Inference response: ", value.map)
        return create(ComputationMessageSchema, {
            nodeId: message.nodeId,
            requestId: message.requestId,
            data: {
                value,
                case: "intermediate"
            }
        });
    }

    invalidateCache(requestId?: string) {
        if (requestId == null) {
            this.pastKeyValues.clear();
        } else {
            this.pastKeyValues.delete(requestId);
        }
    }
}