#!/bin/bash
# Source Vitis/XRT environment
set -e

source /tools/Xilinx/2024.2/Vitis/2024.2/settings64.sh
export XILINX_XRT="/opt/xilinx/xrt"
export LD_LIBRARY_PATH="$XILINX_XRT/lib:$LD_LIBRARY_PATH"

for tool in vitis_hls v++ vivado xrt-smi; do
    command -v $tool &>/dev/null || { echo "Error: $tool not found"; exit 1; }
done

echo "Environment ready."
