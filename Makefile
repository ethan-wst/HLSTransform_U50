# Makefile for HLSTransform - Llama2 FPGA Accelerator

# Paths
PROJECT_ROOT  := $(shell pwd)
ARCHIVES_DIR  := $(PROJECT_ROOT)/archives
BUILD_DIR     := $(PROJECT_ROOT)/build
DATA_DIR      := $(PROJECT_ROOT)/data
SCRIPTS       := $(PROJECT_ROOT)/scripts
CONFIGS_DIR   := $(PROJECT_ROOT)/configs
KERNEL_DIR    := $(PROJECT_ROOT)/src/kernels
HOST_SRC      := $(PROJECT_ROOT)/src/host/llama2_inference.cpp

# Build subdirectories
SIM_DIR       := $(BUILD_DIR)/sim
SYNTH_DIR     := $(BUILD_DIR)/synth
LINK_DIR      := $(BUILD_DIR)/link

# Outputs
HOST_EXE      := $(BUILD_DIR)/llama2_inference_host
KERNEL_XO     := $(BUILD_DIR)/llama2_inference.xo
KERNEL_XCLBIN := $(BUILD_DIR)/llama2_inference.xclbin

# Platform & configs
PLATFORM      := $(PROJECT_ROOT)/platform/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm
HLS_CFG       := $(PROJECT_ROOT)/configs/hls_config.cfg
LINK_CFG      := $(PROJECT_ROOT)/configs/link.cfg

# Tools
VITIS_HLS     ?= vitis_hls
VPP           ?= v++

.PHONY: all help setup csim syn cosim link host run build archive \
        clean clean_csim clean_cosim clean_syn check_env

all: help

help:
	@echo ""
	@echo "HLSTransform Build Targets"
	@echo "============================================"
	@echo "  Simulation (vitis_hls):"
	@echo "    make csim       C simulation"
	@echo "    make cosim      C/RTL co-simulation (requires syn first)"
	@echo ""
	@echo "  Hardware Build (v++):"
	@echo "    make syn        HLS synthesis + package .xo"
	@echo "    make link       Place & route .xo -> .xclbin"
	@echo "    make host       Build host application"
	@echo "    make build      Full build: syn + link + host (with logging)"
	@echo ""
	@echo "  Execution:"
	@echo "    make run        Run on FPGA"
	@echo ""
	@echo "  Utilities:"
	@echo "    make archive    Archive build + sources (timestamped)"
	@echo "    make check_env  Verify tool installation"
	@echo "    make clean      Remove all build outputs"
	@echo "============================================"
	@echo ""

# Setup
setup:
	@mkdir -p $(SIM_DIR)/csim $(SIM_DIR)/cosim $(SYNTH_DIR) $(LINK_DIR)

# C Simulation
csim: setup
	@test -f $(DATA_DIR)/models/weights.bin   || { echo "ERROR: weights.bin not found";   exit 1; }
	@test -f $(DATA_DIR)/models/tokenizer.bin || { echo "ERROR: tokenizer.bin not found"; exit 1; }
	vitis-run --mode hls --csim \
		--config $(HLS_CFG) \
		--work_dir $(SIM_DIR)/csim

# HLS Synthesis
syn: setup
	@test -f $(HLS_CFG)  || { echo "ERROR: $(HLS_CFG) not found"; exit 1; }
	@test -f $(PLATFORM) || { echo "ERROR: $(PLATFORM) not found"; exit 1; }
	$(VPP) --compile --mode hls \
		--config $(HLS_CFG) \
		--work_dir $(SYNTH_DIR)
	@XO=$$(find $(SYNTH_DIR) -name "*.xo" 2>/dev/null | head -1); \
	if [ -z "$$XO" ]; then \
		echo "ERROR: .xo not found in $(SYNTH_DIR) after synthesis"; exit 1; \
	fi; \
	cp "$$XO" $(KERNEL_XO); \
	

# Co-Simulation (Work in Progress)
cosim: syn
	vitis-run --mode hls --cosim \
		--config $(HLS_CFG) \
		--work_dir $(SIM_DIR)/cosim

# v++ Link: Place & Route .xo -> .xclbin
link: syn
	@test -f $(KERNEL_XO) || { echo "ERROR: $(KERNEL_XO) not found. Run 'make syn' first."; exit 1; }
	$(SCRIPTS)/shell/vpp_link.sh \
		$(KERNEL_XO) \
		$(KERNEL_XCLBIN) \
		$(PLATFORM) \
		$(LINK_CFG) \
		$(LINK_DIR)

# Host Application
host: setup
	$(SCRIPTS)/shell/build_host.sh $(HOST_SRC) $(HOST_EXE) $(KERNEL_DIR)

# Full Build: syn -> link -> host (with orchestration logging)
build:
	@test -f $(HLS_CFG)  || { echo "ERROR: $(HLS_CFG) not found";  exit 1; }
	@test -f $(LINK_CFG) || { echo "ERROR: $(LINK_CFG) not found"; exit 1; }
	@test -f $(PLATFORM) || { echo "ERROR: $(PLATFORM) not found"; exit 1; }
	$(SCRIPTS)/shell/full_build.sh

# Run on FPGA
run: host
	@test -f $(KERNEL_XCLBIN)              || { echo "ERROR: Run 'make link' first";  exit 1; }
	@test -f $(DATA_DIR)/models/weights.bin || { echo "ERROR: weights.bin not found"; exit 1; }
	$(HOST_EXE) $(DATA_DIR)/models/weights.bin \
		-z $(DATA_DIR)/models/tokenizer.bin \
		-k $(KERNEL_XCLBIN) \
		-m generate -n 256

# Archive
archive:
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	ARCHIVE_DIR=$(ARCHIVES_DIR)/$$TIMESTAMP; \
	mkdir -p $$ARCHIVE_DIR/src/kernels $$ARCHIVE_DIR/src/host $$ARCHIVE_DIR/configs; \
	cp -r $(BUILD_DIR)/.        $$ARCHIVE_DIR/build/      2>/dev/null || true; \
	cp -r $(CONFIGS_DIR)/.      $$ARCHIVE_DIR/configs/; \
	cp -r $(KERNEL_DIR)/.       $$ARCHIVE_DIR/src/kernels/; \
	cp    $(HOST_SRC)           $$ARCHIVE_DIR/src/host/; \
	cp    $(PROJECT_ROOT)/Makefile $$ARCHIVE_DIR/; \
	ls $(PROJECT_ROOT)/build_*.log 2>/dev/null && \
		cp $(PROJECT_ROOT)/build_*.log $$ARCHIVE_DIR/ || true; \
	echo ">>> Archived to $$ARCHIVE_DIR"

# Clean
clean_csim:
	rm -rf $(SIM_DIR)/csim

clean_cosim:
	rm -rf $(SIM_DIR)/cosim

clean_syn:
	rm -rf $(SYNTH_DIR) $(KERNEL_XO)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PROJECT_ROOT)/build_*.log

# Environment Check
check_env:
	@echo "vitis_hls : $$(which $(VITIS_HLS) 2>/dev/null  || echo 'NOT FOUND')"
	@echo "v++       : $$(which $(VPP) 2>/dev/null         || echo 'NOT FOUND')"
	@echo "XRT       : $$(echo $${XILINX_XRT:-NOT SET})"
	@echo "Vitis     : $$(echo $${XILINX_VITIS:-NOT SET})"
	@echo "Platform  : $(PLATFORM)"
	@test -f $(PLATFORM) || echo "WARNING: Platform file not found"
	@test -f $(HLS_CFG)  || echo "WARNING: HLS config not found: $(HLS_CFG)"
	@test -f $(LINK_CFG) || echo "WARNING: Link config not found: $(LINK_CFG)"