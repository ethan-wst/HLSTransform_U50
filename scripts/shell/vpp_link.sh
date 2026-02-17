#!/bin/bash
# v++ link: Place & Route .xo -> .xclbin

set -e

XO="$1"
XCLBIN="$2"
PLATFORM="$3"
LINK_CFG="$4"
LINK_DIR="$5"

[ -f "$XO" ]       || { echo "Error: $XO not found"; exit 1; }
[ -f "$PLATFORM" ] || { echo "Error: Platform $PLATFORM not found"; exit 1; }
[ -f "$LINK_CFG" ] || { echo "Error: $LINK_CFG not found"; exit 1; }

# If platform is .xpfm, it references hw/hw.xsa in the same directory - must exist
if [[ "$PLATFORM" == *.xpfm ]]; then
    PLATFORM_DIR=$(dirname "$PLATFORM")
    [ -f "$PLATFORM_DIR/hw/hw.xsa" ] || {
        echo "Error: Platform incomplete. Missing $PLATFORM_DIR/hw/hw.xsa"
        exit 1
    }
fi

WORK_DIR="${LINK_DIR}/work"
REPORTS_DIR="${LINK_DIR}/reports"
LOGS_DIR="${LINK_DIR}/logs"

mkdir -p "${WORK_DIR}" "${REPORTS_DIR}" "${LOGS_DIR}"

# Run v++ from work directory to contain outputs
cd "${WORK_DIR}"

v++ --platform "$PLATFORM" \
    --target hw \
    --config "$LINK_CFG" \
    --optimize 3 \
    --save-temps \
    --temp_dir "${WORK_DIR}/temp" \
    --log_dir "${LOGS_DIR}" \
    --report_dir "${REPORTS_DIR}" \
    --link "$XO" \
    -o "$XCLBIN"

# Copy final output to expected location
[ -f "$XCLBIN" ] && cp "$XCLBIN" "${LINK_DIR}/"
