Have a look at related work for:

-   WeInfer: Unleashing the Power of WebGPU on LLM Inference in Web Browsers
-   Large Language Model Partitioning for Low-Latency Inference at the Edge
-   EdgeShard: Efficient LLM Inference via Collaborative Edge Computing

## Summaries

-   Distributed inference and fine-tuning of large language models over the internet  
    https://arxiv.org/abs/2312.08361  
    10/10 - perfectly relevant; lots of details and ideas; dense and technical language

    run LLM inference/fine-tuning distributed to maximize throughput  
    allow for random failures/disconnects  
    figure out partitioning with uneven hardware  
    describes kv-cache rebuilding on disconnects; use path finding algorithms to find best throughput; load balancing

-   EdgeShard: Efficient LLM Inference via Collaborative Edge Computing  
    https://arxiv.org/abs/2405.14371  
    9/10 - great paper; good fit for partitioning, especially pipeline parallelism; good fit for topic

    partitioning of LLMs based on device heterogeneity, bandwidth limitations, and model complexity  
    optimize latency and throughput  
    uses pipeline parallelism to reduce idle time

-   WeInfer: Unleashing the Power of WebGPU on LLM Inference in Web Browsers  
    https://dl.acm.org/doi/abs/10.1145/3696410.3714553  
    9/10 - relevant for performance optimization; technical details; tested extensively; nice reference/related work

    use WebGPU for browser LLM inference  
    optimize WebGPU buffer lifecycle (buffer reuse)  
    decouple resource preparation from GPU execution (async pipeline)  
    modified version of WebLLM

-   LinguaLinked: Distributed Large Language Model Inference on Mobile  
    https://arxiv.org/abs/2312.00388  
    8/10 - highly relevant; focus on resouce contraints; language gives overview rather than detailed descriptions

    distributed LLM inference on mobile devices  
    assing model segments based on device resources  
    load balancing

-   WebLLM: A High-Performance In-Browser LLM Inference Engine  
    https://arxiv.org/abs/2412.15803  
    7/10 - not published; nice reference; method a bit lacking; shows different way of compiling to web instead of ONNX; see WeInfer/TVM papers

    JS library to run MLC-LLM/TVM models in the browser  
    uses WebGPU to achieve up to 80% native performance (WebLLM vs MLC-LLM, so other LLM inference like vLLM might be faster)

-   Adaptive Orchestration for Inference of Large Foundation Models at the Edge / Intelligent Orchestration of Distributed Large Foundation Model Inference at the Edge  
    https://arxiv.org/pdf/2504.03668  
    7/10 - relevant ideas; higher level ideas; formulation takes time to read

    split and distribute large model dynamically  
    QoS-aware and privacy preserving  
    distribute workloads to nodes based on metrics  
    monitoring and profiling, orchestrator, splitting/re-partitioning, broadcast (updates nodes with new partitions)

-   Model Parallelism on Distributed Infrastructure: A Literature Review from Theory to LLM Case-Studies  
    https://arxiv.org/abs/2403.03699  
    7/10 - not yet published; great reference for model parallelism; high level (survey), not to many details

    survey on distributed model parallelism with LLM case study  
    explains types (inter- and intra-operator parallelism) and lists papers for each  
    lists papers for LLMs that use parallelism

-   Large Language Model Partitioning for Low-Latency Inference at the Edge  
    https://arxiv.org/abs/2505.02533  
    6/10 - not yet published; nice reference; strong focus on fine-grained partitioning

    partition and migrate layers and kv cache during inference  
    fine-grained attention-head level partitioning to prevent kv cache from filling device  
    considers memory capacity and migration cost

-   Web-Centric Federated Learning over the Cloud-Edge Continuum Leveraging ONNX and WASM  
    https://ieeexplore.ieee.org/document/10733614  
    5/10 - nice reference, but to different to be technically useful; gives basics of ML in browser

    federated learning using ONNX runtime web  
    easy use, just requires browser and server  
    training on MNIST and CIFAR10

-   TVM: An Automated End-to-End Optimizing Compiler for Deep Learning  
    https://arxiv.org/abs/1802.04799  
    5/10 - great paper/reference; topic to different to be immediately useful

    compiles model to diverse hardware backends while optimizing for said hardware  
    use tensor expression language to represent models; optimize tensor operators using program; rewrite graph for higher level optimizations

-   nnWeb: Towards efficient WebGPU-based DNN inference via automatic collaborative offloading  
    https://www.sciencedirect.com/science/article/pii/S1389128625004566  
    4/10 - possible reference; is journal good?; decent overview and ideas when it comes to client-server partitioning

    dynamic model partitioning between client browser and server  
    optimize splitting for latency  
    built on top of tensorflow.js

-   Adaptive layer splitting for wireless large language model inference in edge computing: a model-based reinforcement learning approach  
    https://link.springer.com/article/10.1631/FITEE.2400468  
    4/10 - relevant if focus is on finding nice splits, otherwise maybe reference; a bit all over, seems to complicated for what it provides

    uses RL to find optimal splitting points in LLMs

-   ServerlessLLM: Low-Latency Serverless Inference for Large Language Models  
    https://www.usenix.org/conference/osdi24/presentation/fu  
    4/10 - great paper; topic too different

    low-latency serverless distributed inference (no model partitioning)  
    checkpoint loading (use disk and RAM on inference server instead of loading from model repo)  
    live migration of inference (migrate token, recompute kv cache)  
    optimized startup times

-   DistML.js: Installation-free Distributed Deep Learning Framework for Web Browsers  
    https://arxiv.org/abs/2407.01023  
    3/10 - not yet published; possibly a reference, otherwise too different

    distributed training and inference in browsers  
    library for ML in browsers  
    uses WebGL

-   SplitLLM: Collaborative Inference of LLMs for Model Placement and Throughput Optimization  
    https://arxiv.org/abs/2410.10759  
    2/10 - not yet published; lots of unnecessary bloat; difficult to read and figure out what's going on

    split model between client and server
