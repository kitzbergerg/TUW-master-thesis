# Roadmap

## Foundation

-   [x] Exploration of technologies: Investigate wasm-nn, WebGPU, ONNX format and other technologies
-   [x] Local model execution: Implement and benchmark a standalone ML model running in WebAssembly
-   [x] Split small model manually. Run in different WASM modules manually. Verify that output is the same as non-split model. Goal: Understand model architectures better
-   [ ] Literature review: Review of distributed inference, WebAssembly performance, and LLM deployment techniques

## Basic distributed system

-   [x] Model splitting: Implement techniques for partitioning large models across multiple clients
-   [x] Web server implementation: Create a system to serve and coordinate WebAssembly modules
-   [x] Basic task distribution: Implement simple task allocation across multiple clients
-   [ ] Performance measurement: Develop metrics and tools for measuring system performance

## Advanced features

-   [ ] Computational Graph Optimization: Develop an execution flow for distributed inference
-   [ ] Caching strategies: Develop and evaluate different approaches to caching for improved performance
-   [x] Multi User: Allow for multiple users to send inference requests at the same time
-   [ ] Multi User: Implement batching

## Evaluation

-   [ ] Case study with LLMs: Implement and evaluate the system using models like Llama 3 or DeepSeek
-   [ ] Comparative analysis: Compare your approach with centralized inference and container-based distributed solutions
-   [ ] Benchmark development: Create reproducible benchmarks for federated inference systems
