#!/bin/bash
# Compile host application
# Args: $1=source $2=output $3=kernel_include_dir
set -e

SRC="$1"
OUT="$2"
KERNEL_INC="$3"

[ -f "$SRC" ] || { echo "Error: $SRC not found"; exit 1; }

g++ -std=c++17 -O3 -Wall \
    -I${XILINX_XRT}/include \
    -I"$KERNEL_INC" \
    "$SRC" \
    -o "$OUT" \
    -L${XILINX_XRT}/lib -lxrt_coreutil -pthread -lrt -lstdc++

echo "Host built: $OUT"
