# Source this on turing1 before llama-completion / llama-cli.
export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$HOME/HGA/Inference/Qwen3_8_27B/third_party/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
export HGA_ROOT="$HOME/HGA/Inference/Qwen3_8_27B"
export HGA_BIN="$HGA_ROOT/third_party/llama.cpp/build/bin"
export HGA_MODEL="${HGA_MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf}"
