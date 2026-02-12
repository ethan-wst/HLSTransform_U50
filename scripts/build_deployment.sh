#!/bin/bash
#==============================================================================
# build_deployment.sh - Full Hardware Build for FPGA Deployment
#==============================================================================
# This script runs the complete HLS → hardware build flow:
#   1. HLS Synthesis (5-15 min)
#   2. Package to .xo + Vitis link to .xclbin (make link does both; 2-6 hours)
#   3. Build host application (1 min)
#
# Usage:
#   ./scripts/build_deployment.sh [log_file]
#
# To run in background (survives disconnect):
#   nohup ./scripts/build_deployment.sh > build.log 2>&1 &
#   tail -f build.log  # monitor progress
#==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Navigate to project root first (needed for log file location)
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Setup logging
if [ $# -ge 1 ]; then
    # If absolute path provided, use it; otherwise make it relative to project root
    if [[ "$1" == /* ]]; then
        LOG_FILE="$1"
    else
        LOG_FILE="${PROJECT_ROOT}/$1"
    fi
else
    # Generate timestamped log file in project root
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${PROJECT_ROOT}/build_${TIMESTAMP}.log"
fi

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Function to log without timestamp (for multi-line output)
log_raw() {
    echo "$*"
}

# Redirect all output to log file (both stdout and stderr)
# This ensures everything is logged even if running with nohup
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Trap to log exit status
trap 'EXIT_CODE=$?; if [ $EXIT_CODE -ne 0 ]; then log "Script exited with error code: $EXIT_CODE"; fi' EXIT ERR

# Record start time
START_TIME=$(date +%s)
log_raw "=========================================="
log "HLSTransform Hardware Build Started"
log "Log file: $LOG_FILE"
log_raw "=========================================="
log_raw ""

# Source Xilinx tools
log "[1/4] Setting up Xilinx environment..."
source /tools/Xilinx/2024.2/Vitis/2024.2/settings64.sh
export XILINX_XRT=/opt/xilinx/xrt
log "✓ Environment configured"
log_raw ""
log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"
log_raw ""

# Step 1: HLS Synthesis
log_raw "=========================================="
log "[2/4] Running HLS Synthesis..."
log "Estimated time: 5-15 minutes"
log_raw "=========================================="
if make syn; then
    log "✓ HLS Synthesis completed"
else
    log "✗ HLS Synthesis failed"
    exit 1
fi
log_raw ""

# Step 2: Package .xo and Vitis link (make link runs xo then v++ link; only runs xo once)
log_raw "=========================================="
log "[3/4] Packaging kernel to .xo and running Vitis v++ link..."
log "Estimated time: 2-6 hours (packaging ~1 min, then place & route)"
log_raw "=========================================="
if make link; then
    log "✓ Vitis link completed - .xclbin generated"
else
    log "✗ Vitis link failed"
    exit 1
fi
log_raw ""

# Step 3: Build Host Application
log_raw "=========================================="
log "[4/4] Building host application..."
log "Estimated time: 1 minute"
log_raw "=========================================="
if make host; then
    log "✓ Host application built"
else
    log "✗ Host build failed"
    exit 1
fi
log_raw ""

# Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

log_raw "=========================================="
log "Build Completed Successfully!"
log_raw "=========================================="
log "End Time: $(date)"
log "Total Time: ${HOURS}h ${MINUTES}m"
log_raw ""
log_raw "Outputs:"
log_raw "  Kernel:  build/vitis/llama2_inference.xclbin"
log_raw "  Host:    build/host/llama2_inference_host"
log_raw ""
log_raw "Next steps:"
log_raw "  1. Verify outputs exist: ls -lh build/vitis/*.xclbin build/host/*_host"
log_raw "  2. Run on FPGA: make run"
log_raw "=========================================="
log_raw ""
log "Full log saved to: $LOG_FILE"
