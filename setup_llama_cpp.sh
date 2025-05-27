#!/bin/bash

# Clone llama.cpp with submodules
git clone --recursive https://github.com/ggerganov/llama.cpp

# Build llama.cpp (CPU only)
cd llama.cpp
mkdir -p build
cd build
cmake .. -DLLAMA_CUBLAS=OFF
cmake --build . --config Release -j

# Move llama-quantize to root for Unsloth compatibility
cd ..
cp build/bin/llama-quantize .

# Verify it exists:
ls llama-quantize