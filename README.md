# HLSTransform: AMD Alveo U50

An extension of HLSTransform Energy-Efficient Llama 2 Inference on FPGAs Via High Level Synthesis [[arXiv:2405.00738]](https://arxiv.org/abs/2405.00738) — He et al., 2024

This work implements and extends the findings of HLSTransform to an AMD Alveo U50 FPGA target, aiming to provide a replicable and optimized implementation for Llama2.c inference on FPGA hardware. Primary modification from the original work are based around the relative decrease in on-chip memory and computational resources of the U50 in comparison to the VU9P hardware.

## Overview

FPGA hardware provides a feasible platform for energy-efficient LLM inference in non-batched settings. This implementation aims to support this application 
by extending the original HLSTransform architecture [[He et al., 2024]](https://arxiv.org/abs/2405.00738), which targeted the Xilinx Virtex UltraScale+ VU9P, to the AMD Alveo U50,
a less cost-prohibitive platform at approximately one-third the cost. This aims to lower the barrier to entry for consumer and edge deployment while retaining  the core energy-efficiency advantages of FPGA-based inference over GPU alternatives.

---

<table>
<tr>
<td valign="top">

## Target Hardware
| Parameter | Value |
|-----------|-------|
| Card | AMD Alveo U50 |
| HBM2 | 8 GB, up to 316 GB/s |
| BRAM | 1,344 × 36Kb blocks (47.3 Mb) |
| URAM | 640 × 288Kb blocks (180.0 Mb) |
| DSP Slices | 5,952 |
| LUTs | 872K |


</td>
<td width="40px"></td>
<td valign="top">

## Model Configuration    
| Parameter | Value |  
|-----------|-------|  
| `dim` | 768 |  
| `hidden_dim` | 2,048 | |
| `vocab_size` | 32,000 |  |
| `n_layers/n_heads` | 12 |
| Quantization | INT8 |
| Group size (`GS`) | 64 |

</td>
</tr>
</table>

## Architecture

### Memory Layout

Due to the limited on-chip memory (BRAM/URAM) of the U50 and FPGA architecture in general, it is necessary to maintain quantized weight matrices and corresponding scales on off-chip HBM memory, this requires the reconfiguration of AXI interfaces, and computational strucuture to insure data transfer burst can be properly infered during Vitis HLS synthesis.

| Data | Location | Notes |
|------|----------|-------|
| Weight matrices | HBM | 512-bit burst, streamed sequentially (future work will pursue parallel computation) |
| Weight scales | HBM → URAM preload | Single burst load per matmul call (exculuding classification due to on-chip memory constraints)|
| Activation vector | On-chip BRAM/URAM | Highly partitioned to allow for parallel access|
| Activation scales | On-chip | Highly partitioned to allow for parallel access |
| KV cache | HBM | Limited infered burst due to strided access (future work will pursue structural modifications) |

---

## Known Issues / In Progress

- **KV cache stride failures** — `key_cache` and `value_cache` AXI strided accesses causing attention II=4 violations. 
  - Layout restructuring planned.
- **Sequential Matmul Calculation** — `outer_loop` of matmul function is unoptimized leading to sequential computation.
  - Parallel computation planned, requires restructuring of weight access and local buffering of weights.
- **Underutilization of Dataflow Capability** — HLS allows for dataflow sections, creating buffers between functions for pipelined execution. 
  - Requires strict adherence to a rule set
  - Major structural modification may be necessary for large scal implementation
  - Limited dataflow additions planned, where implementation is straight forward


## Build System

Modification to MakeFile and corresponding shell scripts may be neccessary depending on exact platform, vitis, and vivado installation type and/or location

TODO: expand on installation instructions, increase configurability of scripts, 

```bash
# Synthesis -> Link -> Host Compilation (~4-6 hrs)
make build

# Run on FPGA
make run

# Archives build artifacts
make archive

# Cleans up build artifacts
make clean
```

Build outputs are written to `build/`. Optional archive of all build artifacts (logs, reports, outputs, etc.), src files (.cpp/.h), and configurations (.cfg) stored in timestamp directories


## References

- [HLSTransform: Energy-Efficient Llama 2 Inference on FPGAs Via High Level Synthesis](https://arxiv.org/abs/2405.00738) — He et al., 2024
- [llama2.c](https://github.com/karpathy/llama2.c) — reference model implementation

