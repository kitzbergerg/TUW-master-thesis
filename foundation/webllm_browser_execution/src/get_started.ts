import * as webllm from "@mlc-ai/web-llm";



let engine: webllm.MLCEngineInterface | undefined = undefined;

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function loadEngine() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };

  const appConfig: webllm.AppConfig = {
    model_list: [
      // TODO: doesn't work, prob because of outdate libs. See https://github.com/mlc-ai/web-llm/issues/675
      {
        model_id: "gemma-3",
        model: window.location.origin + "/models/gemma-3-1b-it-q4f16_1-MLC",
        model_lib: window.location.origin + "/models/gemma-3-1b-it-q4f16_1-webgpu.wasm",
      },
      {
        model_id: "Qwen3",
        model: window.location.origin + "/models/Qwen3-4B-q4f16_1-MLC",
        model_lib: window.location.origin + "/models/Qwen3-4B-q4f16_1-webgpu.wasm",
      },
    ],
  };
  const mlc_engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine("gemma-3", { appConfig: appConfig, initProgressCallback: initProgressCallback },);
  //const mlc_engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine("Llama-3.2-1B-Instruct-q4f32_1-MLC", { initProgressCallback: initProgressCallback },);

  engine = mlc_engine
}

async function runModel() {
  if (engine == undefined) {
    console.log("model not yet loaded")
    return
  }

  const input: HTMLInputElement = document.getElementById("prompt") as HTMLInputElement;
  const prompt: string = input.value;

  console.log("Running model with prompt: " + prompt)
  const request: webllm.ChatCompletionRequest = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [{ role: "user", content: prompt }]
  };


  const asyncChunkGenerator = await engine.chat.completions.create(request);
  let message = "";
  for await (const chunk of asyncChunkGenerator) {
    console.log(chunk);
    message += chunk.choices[0]?.delta?.content || "";
    setLabel("generate-label", message);
    if (chunk.usage) {
      console.log(chunk.usage);
    }
  }
  console.log("Final message:\n", await engine.getMessage());
}

loadEngine();
document.getElementById("run-model-btn")?.addEventListener("click", runModel);
