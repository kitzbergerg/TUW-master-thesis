# TODO

-   Distributed inference
    -   Allow for full chat (generate till end token, handle system/agent/user switch)
    -   Improve performance
        -   onnx graph optimizations
        -   quantization
    -   Split model further (four/eight parts)
-   WebLLM/MLC/TVM
    -   Compile and run non-standard model
    -   Run split model?
-   Transformers.js
    -   Explore how transformers uses ONNX runtime, it seems to use webnn, figure out who that compares to webgpu
    -   Compile and run non-standard model
    -   Run split model?
-   Decision: [inference-framework](./decisions/1.1-inference-framework.md)
