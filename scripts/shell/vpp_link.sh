#!/bin/bash
# v++ link: Place & Route .xo -> .xclbin
# Args: $1=kernel.xo $2=output.xclbin $3=platform $4=link.cfg
set -e

XO="$1"
XCLBIN="$2"
PLATFORM="$3"
LINK_CFG="$4"

[ -f "$XO" ]       || { echo "Error: $XO not found"; exit 1; }
[ -f "$PLATFORM" ] || { echo "Error: Platform $PLATFORM not found"; exit 1; }
[ -f "$LINK_CFG" ] || { echo "Error: $LINK_CFG not found"; exit 1; }

# If platform is .xpfm, it references hw/hw.xsa in the same directory - must exist
if [[ "$PLATFORM" == *.xpfm ]]; then
    PLATFORM_DIR=$(dirname "$PLATFORM")
    if [[ ! -f "$PLATFORM_DIR/hw/hw.xsa" ]]; then
        echo "Error: Platform is incomplete. $PLATFORM references hw/hw.xsa but it is missing."
        echo "  Expected: $PLATFORM_DIR/hw/hw.xsa"
        echo ""
        echo "Download the full Alveo U50 platform (Development Target Platform) from:"
        echo "  https://www.amd.com/en/support/downloads/alveo-downloads.html"
        echo "Then extract it so that platform/hw/hw.xsa exists next to the .xpfm file."
        exit 1
    fi
fi

echo "Linking $XO -> $XCLBIN"
echo "WARNING: This takes 2-6 hours"

v++ --platform "$PLATFORM" \
    --target hw \
    --config "$LINK_CFG" \
    --optimize 3 \
    --save-temps \
    --link "$XO" \
    -o "$XCLBIN"

echo "XCLBIN generated: $XCLBIN"
