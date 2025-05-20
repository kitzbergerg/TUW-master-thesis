# TODO

-   Distributed inference
    -   Improve performance
        -   better (de)serialization (sending json is inefficient)
        -   onnx graph optimizations
        -   quantization
    -   Handle connecting/disconnecting clients
        -   rebuild cache on different node in case of disconnect
        -   use extra clients instead of just 1 client per node
    -   Split model further (two/four/eight parts)
-   WebLLM/MLC/TVM
    -   Compile and run non-standard model
    -   Run split model?
-   Transformers.js
    -   Compile and run non-standard model
    -   Run split model?
    -   Explore how transformers uses ONNX runtime
-   Decision: [inference-framework](./decisions/1.1-inference-framework.md)
