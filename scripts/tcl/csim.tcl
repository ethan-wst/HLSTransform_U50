# C Simulation - functional verification of kernel
# Run from: build/csim/
set src_dir "../../src/kernels"
set tb_dir  "../../src/testbench"

open_project -reset csim_project
add_files $src_dir/forward.cpp
add_files $src_dir/forward.h
add_files $src_dir/config.h
add_files $src_dir/typedefs.h
add_files -tb $tb_dir/tb_forward.cpp
set_top forward
open_solution "solution1"
set_part {xcu50-fsvh2104-2-e}
csim_design
exit
