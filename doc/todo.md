# TODO

-   Distributed inference
    -   Figure out why ollama phi generates better answers than web (chat template?, system prompt?)
    -   Allow for full chat (generate till end token, handle system/agent/user switch)
    -   Improve performance
        -   onnx graph optimizations
        -   quantization
    -   Split model further (four/eight parts)
-   WebLLM/MLC/TVM
    -   Compile and run non-standard model
    -   Run split model?
-   Transformers.js
    -   Compile and run non-standard model
    -   Run split model?
    -   Explore how transformers uses ONNX runtime
-   Decision: [inference-framework](./decisions/1.1-inference-framework.md)
