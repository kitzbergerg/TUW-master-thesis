# WebLLM

## Setup

On linux this requires enabling some flags for chrome (for firefox only nightly works):

-   chrome://flags/#enable-unsafe-webgpu
-   chrome://flags/#enable-vulkan

For faster inference start chrome with:

-   --enable-dawn-features=disable_robustness

## Compiling models

Build tvm:

https://llm.mlc.ai/docs/install/tvm.html#option-2-build-from-source
https://tvm.apache.org/docs/install/from_source.html

Build mlc from source:

https://llm.mlc.ai/docs/install/mlc_llm.html#option-2-build-from-source

Compile model:

https://llm.mlc.ai/docs/compilation/compile_models.html

Remember to activate your conda env `conda activate model-conversion`.
