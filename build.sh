#!/usr/bin/env bash
set -euo pipefail

ARCH=${ARCH:-sm_89}
FLAGS="-arch=${ARCH} -O2"

nvcc $FLAGS fused_example.cu    -o fused_example
nvcc $FLAGS fused_zo_example.cu -o fused_zo_example
