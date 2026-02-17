#!/bin/bash
# Full build orchestration: syn -> link -> host -> archive

set -e
set -u

# Navigate to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# Setup logging
if [ $# -ge 1 ]; then
    if [[ "$1" == /* ]]; then
        LOG_FILE="$1"
    else
        LOG_FILE="${PROJECT_ROOT}/$1"
    fi
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${PROJECT_ROOT}/build_${TIMESTAMP}.log"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}
log_raw() {
    echo "$*"
}

exec > >(tee -a "$LOG_FILE")
exec 2>&1

trap 'EXIT_CODE=$?; [ $EXIT_CODE -ne 0 ] && log "FAILED with exit code: $EXIT_CODE"' EXIT
START_TIME=$(date +%s)

# Environment
log_raw ""
log "[1/4] Setting up Xilinx environment..."
source /tools/Xilinx/2024.2/Vitis/2024.2/settings64.sh
export XILINX_XRT="/opt/xilinx/xrt"
export LD_LIBRARY_PATH="$XILINX_XRT/lib:$LD_LIBRARY_PATH"
for tool in vitis_hls v++ vivado xrt-smi; do
    command -v $tool &>/dev/null || { echo "Error: $tool not found"; exit 1; }
done
log_raw ""

# HLS Synthesis
log "[2/4] HLS Synthesis (.xo)"
log_raw ""

if make syn; then
    log "Synthesis complete: $(ls -sh build/llama2_inference.xo 2>/dev/null || echo 'xo generated')"
else
    log "Synthesis failed"; exit 1
fi
log_raw ""

# v++ Link
log_raw ""
log     "[3/4] v++ Link (.xo -> .xclbin)"
log_raw ""
if make link; then
    log "Link complete: $(ls -sh $(PROJECT_ROOT)/build/llama2_inference.xclbin 2>/dev/null || echo 'xclbin generated')"
else
    log "Link failed"; exit 1
fi
log_raw ""

# Host Application
log_raw ""
log     "[4/4] Host Application Build"
log_raw ""
if make host; then
    log "Host built: $(ls -sh $(PROJECT_ROOT)/build/llama2_inference_host 2>/dev/null || echo 'host built')"
else
    log "Host build failed"; exit 1
fi
log_raw ""

# Archive
if make archive; then
    log "Archive created"
else
    log "Archive failed (non-fatal)"; 
fi