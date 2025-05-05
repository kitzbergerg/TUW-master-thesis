import { pipeline, env } from '@huggingface/transformers';


env.localModelPath = '/model/';
env.allowLocalModels = true;
env.allowRemoteModels = false;

async function runModel(text) {
    const generator = await pipeline('text-generation', 'gpt2', { device: "webgpu" },);
    return await generator(text, {
        max_new_tokens: 100,
        repetition_penalty: 1.5, // needs to be > 1 otherwise it outputs garbage
    });
}
async function startModel() {
    const prompt = document.getElementById("prompt").value;
    const result = await runModel(prompt);
    console.log(result)
    const text = result[0].generated_text
    console.log(text)

    document.getElementById("pre").innerHTML = text;
}
window.startModel = startModel
