#!/bin/bash
# Extract key metrics from HLS synthesis report
# Args: $1=report file path
set -e

RPT="$1"

if [ ! -f "$RPT" ]; then
    echo "No report found at: $RPT"
    echo "Run 'make syn' first."
    exit 1
fi

echo "=== Timing ==="
grep -A 15 "== Timing" "$RPT" 2>/dev/null || true
echo ""
echo "=== Utilization ==="
grep -A 25 "== Utilization Estimates" "$RPT" 2>/dev/null || true
echo ""
echo "=== Interface ==="
grep -A 15 "== Interface" "$RPT" 2>/dev/null || true
echo ""
echo "Full report: $RPT"
