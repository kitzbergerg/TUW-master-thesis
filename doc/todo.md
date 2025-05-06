# TODO

-   WebLLM
    -   Compile and run non-standard model
    -   Split model and run split model using webllm (if not possible use MLC/TVM directly)
-   Transformers.js
    -   Compile and run non-standard model
    -   Run split model
    -   Explore how transformers uses ONNX runtime
-   ONNX runtime web
    -   Test other/larger models.
    -   Especially GPU part, it might be difficult to run LLMs using webgpu/webnn efficiently.
-   Run gemma3 on all variants.
-   Run split gemma3 on variants where this is possible.
-   Decision: [inference-framework](./decisions/1.1-inference-framework.md)
