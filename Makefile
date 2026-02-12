# Makefile for HLSTransform - Llama2 FPGA Accelerator


# Paths
PROJECT_ROOT := $(shell pwd)
BUILD_DIR    := $(PROJECT_ROOT)/build
DATA_DIR     := $(PROJECT_ROOT)/data
SCRIPTS      := $(PROJECT_ROOT)/scripts
KERNEL_DIR   := $(PROJECT_ROOT)/src/kernels
HOST_SRC     := $(PROJECT_ROOT)/src/host/llama2_inference.cpp

# Build subdirectories
CSIM_DIR     := $(BUILD_DIR)/csim
HLS_DIR      := $(BUILD_DIR)/hls
VITIS_DIR    := $(BUILD_DIR)/vitis
HOST_DIR     := $(BUILD_DIR)/host

# Outputs
HOST_EXE     := $(HOST_DIR)/llama2_inference_host
KERNEL_XO    := $(VITIS_DIR)/llama2_inference.xo
KERNEL_XCLBIN := $(VITIS_DIR)/llama2_inference.xclbin
SYN_RPT      := $(HLS_DIR)/llama2_inference/solution1/syn/report/llama2_inference_csynth.rpt

# Platform
PLATFORM     := $(PROJECT_ROOT)/platform/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm
LINK_CFG     := $(PROJECT_ROOT)/configs/link.cfg

# Tools
VITIS_HLS    ?= vitis_hls

# ============================================================================
.PHONY: all help setup csim syn cosim xo link host run reports \
        clean clean_csim clean_hls check_env

all: csim

help:
	@echo "HLSTransform Build Targets"
	@echo ""
	@echo "  make csim      C simulation"
	@echo "  make syn       HLS synthesis"
	@echo "  make cosim     C/RTL co-simulation"
	@echo "  make reports   View synthesis reports"
	@echo ""
	@echo "  make xo        Package kernel to .xo"
	@echo "  make link      v++ link to .xclbin"
	@echo "  make host      Build host application"
	@echo "  make run       Run on FPGA"
	@echo ""
	@echo "  make clean     Remove all build outputs"
	@echo "  make check_env Verify tool installation"

# Setup
setup:
	@mkdir -p $(CSIM_DIR) $(HLS_DIR) $(VITIS_DIR) $(HOST_DIR)

# C Simulation
csim: setup
	@test -f $(DATA_DIR)/models/weights.bin    || { echo "ERROR: data/models/weights.bin not found"; exit 1; }
	@test -f $(DATA_DIR)/models/tokenizer.bin  || { echo "ERROR: data/models/tokenizer.bin not found"; exit 1; }
	cd $(CSIM_DIR) && PROJECT_ROOT=$(PROJECT_ROOT) $(VITIS_HLS) -f $(SCRIPTS)/tcl/csim.tcl

# HLS Synthesis
syn: setup
	cd $(HLS_DIR) && PROJECT_ROOT=$(PROJECT_ROOT) $(VITIS_HLS) -f $(SCRIPTS)/tcl/syn.tcl

# Co-Simulation
cosim: syn
	cd $(HLS_DIR) && PROJECT_ROOT=$(PROJECT_ROOT) $(VITIS_HLS) -f $(SCRIPTS)/tcl/cosim.tcl

# Package .xo
xo: syn
	@mkdir -p $(VITIS_DIR)
	cd $(HLS_DIR) && XO_OUTPUT=$(KERNEL_XO) $(VITIS_HLS) -f $(SCRIPTS)/tcl/package_xo.tcl

# v++ Link
link: xo
	$(SCRIPTS)/shell/vpp_link.sh $(KERNEL_XO) $(KERNEL_XCLBIN) $(PLATFORM) $(LINK_CFG)

# Host
host: setup
	$(SCRIPTS)/shell/build_host.sh $(HOST_SRC) $(HOST_EXE) $(KERNEL_DIR)

# Run
run: host
	@test -f $(KERNEL_XCLBIN)              || { echo "ERROR: Run 'make link' first"; exit 1; }
	@test -f $(DATA_DIR)/models/weights.bin || { echo "ERROR: weights.bin not found"; exit 1; }
	$(HOST_EXE) $(DATA_DIR)/models/weights.bin \
		-z $(DATA_DIR)/models/tokenizer.bin \
		-k $(KERNEL_XCLBIN) \
		-m generate -n 256

# Reports
reports:
	@$(SCRIPTS)/shell/reports.sh $(SYN_RPT)

# Clean
clean_csim:
	rm -rf $(CSIM_DIR)

clean_hls:
	rm -rf $(HLS_DIR)

clean:
	rm -rf $(BUILD_DIR)

# Check Environment
check_env:
	@echo "vitis_hls: $$(which $(VITIS_HLS) 2>/dev/null || echo 'NOT FOUND')"
	@echo "v++:       $$(which v++ 2>/dev/null || echo 'NOT FOUND')"
	@echo "XRT:       $$XILINX_XRT"
	@echo "Platform:  $(PLATFORM)"
	@test -f $(PLATFORM) || echo "WARNING: Platform file not found"
