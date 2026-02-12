/**
 * @file typedefs.h
 * @brief Core type definitions for Llama 2 transformer
 */

#pragma once
#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <stdint.h>
#include <stdio.h>

struct Config
{
  int dim;        // transformer dimension
  int hidden_dim; // ffn layers
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;    // max sequence length
  int GS;         // group size for quantization
};

#endif // TYPEDEFS_H