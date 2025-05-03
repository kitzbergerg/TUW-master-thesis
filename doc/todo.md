# TODO

-   Model Conversion and Splitting
    -   Convert gemma3 to onnx and split
    -   Convert llama4 to onnx and split
-   WebLLM
    -   Run basic model using webllm
    -   Compile and run non-standard model
    -   Split model and run split model using webllm (if not possible use MLC directly)
-   Framework and Backend Exploration
    -   Explore onnxruntime-web in more detail. Especially GPU part, it might be difficult to run LLMs using webgpu/webnn efficiently.
    -   Understand WebGPU strengths/weaknesses for LLMs in the browser?
    -   Understand WebNN. What are tradeoffs? It is very experimental, is it even usable?
-   Decision: [inference-framework](./decisions/1.1-inference-framework.md)
    -   What inference frameworks are there other than onnxruntime-web?
    -   How do they compare?
    -   How can I split models and run them there?
    -   webllm
