#!/bin/bash
# Setup environment for HLSTransform

set -e

VITIS_SETTINGS="/tools/Xilinx/2024.2/Vitis/2024.2/settings64.sh"

if [ ! -f "$VITIS_SETTINGS" ]; then
    echo "Error: Vitis not found at $VITIS_SETTINGS"
    exit 1
fi

source "$VITIS_SETTINGS"
export XILINX_XRT="/opt/xilinx/xrt"
export LD_LIBRARY_PATH="$XILINX_XRT/lib:$LD_LIBRARY_PATH"

# Verify tools
for tool in vitis_hls v++ vivado xrt-smi; do
    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool not found"
        exit 1
    fi
done

[ -f "platform/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm" ] || echo "Warning: Platform file not found"
echo "Environment ready. Run 'make help' for build targets."
