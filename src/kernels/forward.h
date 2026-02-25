/**
 * @file forward.h
 * @brief Top-level forward function for Llama2.c transformer
 */

#pragma once

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <stdint.h>

#include "typedefs.h"
#include "config.h"

// Wide integer type for data packing, optimizing burst memory access
// PACK_SIZE must equal casted wide_int8_t size / 8 (ei. number of 8-bit chunks in the wide_int8_t)
typedef ap_uint<512> wide_int8_t;
#define PACK_SIZE 64

extern "C" void forward(
    // Embedding Weights
    float *token_embedding_table,     
    
    // Weights
    int8_t *wq_weights,
    float *wq_scales,
    int8_t *wk_weights,
    float *wk_scales,
    int8_t *wv_weights,
    float *wv_scales,
    int8_t *wo_weights,
    float *wo_scales,
    int8_t *w1_weights,
    float *w1_scales,
    int8_t *w2_weights,
    float *w2_scales,
    int8_t *w3_weights,
    float *w3_scales,
    
    // RMS Weights
    float *rms_att_weight,
    float *rms_ffn_weight,
    float *rms_final_weight,
    
    // Final Classifier Weights
    int8_t *wcls_weights,
    float *wcls_scales,
    
    float *key_cache,
    float *value_cache,
    
    float *out,
    int token,
    int pos
);

template<int D, int N>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws) {

    wide_int8_t *wq_wide = (wide_int8_t *)wq;
    const int NUM_GROUPS = N / GS;

    output_loop: 
    for (int i = 0; i < D; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=768 max=32000
        
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma HLS ARRAY_PARTITION variable=acc complete

        // Calculate group and base wide index once
        int group_offset = i * NUM_GROUPS;
        int base_wide = i * N  / PACK_SIZE;
        
        inner_loop: 
        for (int j = 0; j < N; j+=GS) {
        #pragma HLS PIPELINE II=1

            float partial_sum = 0.0f;

            chunk_loop:
            for (int chunk = 0; chunk < GS; chunk += PACK_SIZE) {
                #pragma HLS UNROLL

                int global_j = j + chunk;
                wide_int8_t w_chunk = wq_wide[base_wide + global_j / PACK_SIZE];
            
                int32_t dot_acc = 0;

                dot_loop: 
                for (int k = 0; k < PACK_SIZE; k++) {
                    #pragma HLS UNROLL

                    int32_t w_val = ((int32_t)((int8_t)w_chunk.range(8*k + 7, 8*k)));
                    int32_t x_val = ((int32_t)xq[global_j + k]);

                    dot_acc = w_val * x_val + dot_acc;
                }
                
                partial_sum += ((float)dot_acc) * ws[i * N /GS + global_j / GS] * xs[global_j / GS];
            }
            acc[(j / GS) % 8] = acc[(j / GS) % 8] + partial_sum;
        }
        
        float sum0 = acc[0] + acc[1];
        float sum1 = acc[2] + acc[3];
        float sum2 = acc[4] + acc[5];
        float sum3 = acc[6] + acc[7];
        xout[i] = (sum0 + sum1) + (sum2 + sum3);
    }
}


template<int MAXSIZE>
void softmax(float *x, int size) {
    #pragma HLS INLINE off

    // Find max
    float max_val = x[0];

    find_max:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXSIZE

        max_val = (x[i] > max_val) ? x[i] : max_val;
    }
    
    // Exp and sum
    float sum = 0.0f;
    exp_sum:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXSIZE

        float exp_val = hls::expf(x[i] - max_val);
        x[i] = exp_val;
        sum = exp_val + sum;
    }

    float inv_sum = 1.0f / sum;
    
    // Normalize
    normalize:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXSIZE
        x[i] *= inv_sum;
    }
}

template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
    #pragma HLS INLINE off

    // Calculate sum of squares
    float ss = 0.0f;

    sum_squares:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=S max=S

        float val = x[j];
        ss = val * val + ss; // Read once, use twice
    }

    ss /= S;
    ss = 1e-5f + ss;
    ss = 1.0f / hls::sqrt(ss);
    
    // Normalize and scale
    normalize:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=S max=S

        o[j] = weight[j] * (ss * x[j]);
    }
}

// TODO: Clean Up
template<int S>
void quantize(int8_t qx_q[S], float qx_s[S/GS], float x[S]) {
    #pragma HLS INLINE off
    
    constexpr int num_groups = S / GS;
    constexpr float inv_Q_MAX = 1.0f / 127.0f;
    
    main_loop:
    for (int group = 0; group < num_groups; group++) {
        #pragma HLS LOOP_TRIPCOUNT min=12 max=32

        // Find max absolute value in group
        float wmax = 0.0f;
        int base_idx = group * GS;

        find_max:
        for (int i = 0; i < GS; i++) {
            #pragma HLS PIPELINE II=1

            float abs_val = hls::fabsf(x[base_idx + i]);
            wmax = (abs_val > wmax) ? abs_val : wmax;
        }
        
        // Calculate scale and quantize
        float scale = wmax * inv_Q_MAX;
        qx_s[group] = scale;
        
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        
        quantize_group:
        for (int i = 0; i < GS; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=GS max=GS

            float quant_val = x[group * GS + i] * inv_scale;
            qx_q[group * GS + i] = (int8_t)quant_val;
        }
    }
}
