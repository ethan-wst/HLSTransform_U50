# Package synthesized kernel to Xilinx Object (.xo)
# Run from: build/hls/ (requires prior synthesis)
# Expects XO_OUTPUT env var set by Makefile
set xo_path $::env(XO_OUTPUT)
open_project forward
open_solution "solution1"
config_export -format xo -output $xo_path
export_design
exit
