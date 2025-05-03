# WebLLM

## Setup

On linux this requires enabling some flags for chrome (for firefox only nightly works):

-   chrome://flags/#enable-unsafe-webgpu
-   chrome://flags/#enable-vulkan

For faster inference start chrome with:

-   --enable-dawn-features=disable_robustness

## Compiling models

Resources:

https://llm.mlc.ai/docs/install/emcc.html#step-2-set-tvm-source-dir-and-mlc-llm-source-dir
https://tvm.apache.org/docs/install/from_source.html
https://llm.mlc.ai/docs/install/mlc_llm.html#option-2-build-from-source
https://llm.mlc.ai/docs/compilation/compile_models.html

### Build mlc

```bash
conda env remove -n mlc-chat-venv
rm -rf python && git checkout HEAD -- python
rm -rf build

conda create -n mlc-chat-venv -c conda-forge \
    "cmake>=3.24,<4" \
    rust \
    git \
    python=3.11
conda activate mlc-chat-venv
export TVM_SOURCE_DIR=`pwd`/3rdparty/tvm
export MLC_LLM_SOURCE_DIR=`pwd`
source ../emsdk/emsdk_env.sh
./web/prep_emcc_deps.sh
mkdir build && cd build

python ../cmake/gen_cmake_config.py

cmake .. && cmake --build . --parallel $(nproc) && cd ..
```

### Build tvm

```bash
conda env remove -n tvm-build-venv
rm -rf python && git checkout HEAD -- python
rm -rf build

conda create -n tvm-build-venv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24,<4" \
    git \
    Cython \
    python=3.11
conda activate tvm-build-venv
mkdir build && cd build

cp ../cmake/config.cmake .
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
echo "set(USE_CUDA   OFF)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake
echo "set(USE_FLASHINFER OFF)" >> config.cmake
echo "set(FLASHINFER_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake
echo "set(CMAKE_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake
echo "set(USE_ROCM ON)" >> config.cmake

cmake .. && cmake --build . --parallel $(nproc) && cd ..
```

### Compile model

```bash
conda create -n myenv -c conda-forge python=3.11
conda activate myenv

CURRENT=`pwd`
cd mlc-llm/python && pip install -e .
cd mlc-llm/3rdparty/tvm/python && pip install -e .
cd $CURRENT

MODEL_NAME=gemma-3-1b-it
CONV_TEMPLATE=gemma3_instruction
mlc_llm convert_weight ./dist/models/$MODEL_NAME/ --quantization q4f16_1 -o dist/$MODEL_NAME-q4f16_1-MLC
mlc_llm gen_config ./dist/models/$MODEL_NAME/ --quantization q4f16_1 --conv-template $CONV_TEMPLATE -o dist/$MODEL_NAME-q4f16_1-MLC/
mlc_llm compile ./dist/$MODEL_NAME-q4f16_1-MLC/mlc-chat-config.json --device webgpu -o dist/libs/$MODEL_NAME-q4f16_1-webgpu.wasm
```
