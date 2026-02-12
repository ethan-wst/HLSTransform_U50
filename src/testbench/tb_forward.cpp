// Testbench for C simulation (functional verification only).
// Loads model weights, runs a short greedy generation loop, prints output.
#include "../kernels/forward.h"
#include "../kernels/config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {
const char *kDefaultPrompt = "I am happy";
constexpr int kSteps = 15;

std::string get_data_path(const char *filename) {
    const char *root = getenv("PROJECT_ROOT");
    std::string base = root ? std::string(root) + "/data/" : "data/";
    return base + filename;
}
} // namespace

struct TokenIndex {
    char *str;
    int id;
};

struct Tokenizer {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
};

struct Weights {
    // Embedding (dequantized for host use).
    float *token_embedding_table;

    // Attention weights (quantized).
    int8_t *wq_weights;
    float *wq_scales;
    int8_t *wk_weights;
    float *wk_scales;
    int8_t *wv_weights;
    float *wv_scales;
    int8_t *wo_weights;
    float *wo_scales;

    // FFN weights (quantized).
    int8_t *w1_weights;
    float *w1_scales;
    int8_t *w2_weights;
    float *w2_scales;
    int8_t *w3_weights;
    float *w3_scales;

    // RMS norm weights (float).
    float *rms_att_weight;
    float *rms_ffn_weight;
    float *rms_final_weight;

    // Classifier weights (quantized).
    int8_t *wcls_weights;
    float *wcls_scales;

    // Track if classifier is shared to avoid double-free.
    bool cls_is_shared;
};

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    
    FILE *file = fopen(tokenizer_path.c_str(), "rb");
    if (!file) { 
        std::cerr << "Failed to open tokenizer: " << tokenizer_path << std::endl;
        exit(1); 
    }
    
    fread(&t->max_token_length, sizeof(int), 1, file);
    
    int len;
    for (int i = 0; i < vocab_size; i++) {
        fread(t->vocab_scores + i, sizeof(float), 1, file);
        fread(&len, sizeof(int), 1, file);
        t->vocab[i] = (char *)malloc(len + 1);
        fread(t->vocab[i], len, 1, file);
        t->vocab[i][len] = '\0';
    }
    
    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    if (t->sorted_vocab) free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return (char *)"[INVALID]";
    char *piece = t->vocab[token];
    // Following BOS (1) token, sentencepiece decoder strips any leading whitespace
    if (prev_token == 1 && piece[0] == ' ') piece++;

    unsigned char byte_val;
    // If this is a raw byte token, map it to the corresponding byte piece
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, 
                                            sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { 
        std::cerr << "NULL text" << std::endl;
        exit(1); 
    }
    
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    
    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 3));
    size_t str_len = 0;
    *n_tokens = 0;
    
    if (bos) tokens[(*n_tokens)++] = 1;
    
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }
    
    while (true) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        
        for (int i = 0; i < (*n_tokens - 1); i++) {
            snprintf(str_buffer, t->max_token_length * 2 + 1 + 2, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        
        if (best_idx == -1) break;
        
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) 
            tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) 
        if (x[i] > max_val) max_val = x[i];
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) 
        x[i] /= sum;
}

// ============================================================================
// Read Checkpoint with Flattened Weights
// ============================================================================
void read_checkpoint(const std::string &checkpoint, Weights *weights) {
    FILE *file = fopen(checkpoint.c_str(), "rb");
    if (!file) { 
        fprintf(stderr, "Couldn't open %s\n", checkpoint.c_str()); 
        exit(1); 
    }
    
    // Read and validate header
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    if (magic != 0x616b3432) { 
        fprintf(stderr, "Bad magic number\n"); 
        exit(1); 
    }
    
    int version;
    fread(&version, sizeof(int), 1, file);
    if (version != 2) { 
        fprintf(stderr, "Unsupported version %d\n", version); 
        exit(1); 
    }
    
    // Read config (skip, already have it from config.h)
    Config temp_config;
    int header_size = 256;
    fread(&temp_config, sizeof(Config) - sizeof(int), 1, file);
    
    uint8_t shared_classifier;
    fread(&shared_classifier, sizeof(uint8_t), 1, file);
    
    int group_size;
    fread(&group_size, sizeof(int), 1, file);
    
    // Seek to start of weights
    fseek(file, header_size, SEEK_SET);
    
    // Allocate RMS norm weights
    posix_memalign((void**)&weights->rms_att_weight, 4096, n_layers * dim * sizeof(float));
    posix_memalign((void**)&weights->rms_ffn_weight, 4096, n_layers * dim * sizeof(float));
    posix_memalign((void**)&weights->rms_final_weight, 4096, dim * sizeof(float));
    
    // Read RMS norm weights
    fread(weights->rms_att_weight, sizeof(float), n_layers * dim, file);
    fread(weights->rms_ffn_weight, sizeof(float), n_layers * dim, file);
    fread(weights->rms_final_weight, sizeof(float), dim, file);
    
    // Token embedding - read quantized, then dequantize
    int emb_size = vocab_size * dim;
    int8_t *q_emb;
    float *s_emb;
    posix_memalign((void**)&q_emb, 4096, emb_size * sizeof(int8_t));
    posix_memalign((void**)&s_emb, 4096, (emb_size / GS) * sizeof(float));
    
    fread(q_emb, sizeof(int8_t), emb_size, file);
    fread(s_emb, sizeof(float), emb_size / GS, file);
    
    // Dequantize embedding for host
    posix_memalign((void**)&weights->token_embedding_table, 4096, emb_size * sizeof(float));
    for (int i = 0; i < emb_size; i++) {
        weights->token_embedding_table[i] = q_emb[i] * s_emb[i / GS];
    }
    
    // Attention weights (all layers)
    int att_size = n_layers * dim * dim;
    int att_scale_size = att_size / GS;
    
    posix_memalign((void**)&weights->wq_weights, 4096, att_size * sizeof(int8_t));
    posix_memalign((void**)&weights->wq_scales, 4096, att_scale_size * sizeof(float));
    posix_memalign((void**)&weights->wk_weights, 4096, att_size * sizeof(int8_t));
    posix_memalign((void**)&weights->wk_scales, 4096, att_scale_size * sizeof(float));
    posix_memalign((void**)&weights->wv_weights, 4096, att_size * sizeof(int8_t));
    posix_memalign((void**)&weights->wv_scales, 4096, att_scale_size * sizeof(float));
    posix_memalign((void**)&weights->wo_weights, 4096, att_size * sizeof(int8_t));
    posix_memalign((void**)&weights->wo_scales, 4096, att_scale_size * sizeof(float));
    
    // Read attention weights
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * dim;
        fread(&weights->wq_weights[offset], sizeof(int8_t), dim * dim, file);
        fread(&weights->wq_scales[offset / GS], sizeof(float), dim * dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * dim;
        fread(&weights->wk_weights[offset], sizeof(int8_t), dim * dim, file);
        fread(&weights->wk_scales[offset / GS], sizeof(float), dim * dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * dim;
        fread(&weights->wv_weights[offset], sizeof(int8_t), dim * dim, file);
        fread(&weights->wv_scales[offset / GS], sizeof(float), dim * dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * dim;
        fread(&weights->wo_weights[offset], sizeof(int8_t), dim * dim, file);
        fread(&weights->wo_scales[offset / GS], sizeof(float), dim * dim / GS, file);
    }
    
    // FFN weights (all layers)
    int ffn1_size = n_layers * dim * hidden_dim;
    int ffn1_scale_size = ffn1_size / GS;
    int ffn2_size = n_layers * hidden_dim * dim;
    int ffn2_scale_size = ffn2_size / GS;
    
    posix_memalign((void**)&weights->w1_weights, 4096, ffn1_size * sizeof(int8_t));
    posix_memalign((void**)&weights->w1_scales, 4096, ffn1_scale_size * sizeof(float));
    posix_memalign((void**)&weights->w2_weights, 4096, ffn2_size * sizeof(int8_t));
    posix_memalign((void**)&weights->w2_scales, 4096, ffn2_scale_size * sizeof(float));
    posix_memalign((void**)&weights->w3_weights, 4096, ffn1_size * sizeof(int8_t));
    posix_memalign((void**)&weights->w3_scales, 4096, ffn1_scale_size * sizeof(float));
    
    // Read FFN weights
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * hidden_dim;
        fread(&weights->w1_weights[offset], sizeof(int8_t), dim * hidden_dim, file);
        fread(&weights->w1_scales[offset / GS], sizeof(float), dim * hidden_dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * hidden_dim * dim;
        fread(&weights->w2_weights[offset], sizeof(int8_t), hidden_dim * dim, file);
        fread(&weights->w2_scales[offset / GS], sizeof(float), hidden_dim * dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * hidden_dim;
        fread(&weights->w3_weights[offset], sizeof(int8_t), dim * hidden_dim, file);
        fread(&weights->w3_scales[offset / GS], sizeof(float), dim * hidden_dim / GS, file);
    }
    
    // Classifier weights
    int cls_size = vocab_size * dim;
    int cls_scale_size = cls_size / GS;
    
    if (!shared_classifier) {
        posix_memalign((void**)&weights->wcls_weights, 4096, cls_size * sizeof(int8_t));
        posix_memalign((void**)&weights->wcls_scales, 4096, cls_scale_size * sizeof(float));
        
        fread(weights->wcls_weights, sizeof(int8_t), cls_size, file);
        fread(weights->wcls_scales, sizeof(float), cls_scale_size, file);
        
        weights->cls_is_shared = false;
        
        // Free temporary embedding quantized data
        free(q_emb);
        free(s_emb);
    } else {
        // Reuse token embedding weights (keep quantized version)
        weights->wcls_weights = q_emb;
        weights->wcls_scales = s_emb;
        weights->cls_is_shared = true;
    }
    
    fclose(file);
}

void free_weights(Weights *weights) {
    free(weights->token_embedding_table);
    
    free(weights->wq_weights);
    free(weights->wq_scales);
    free(weights->wk_weights);
    free(weights->wk_scales);
    free(weights->wv_weights);
    free(weights->wv_scales);
    free(weights->wo_weights);
    free(weights->wo_scales);
    
    free(weights->w1_weights);
    free(weights->w1_scales);
    free(weights->w2_weights);
    free(weights->w2_scales);
    free(weights->w3_weights);
    free(weights->w3_scales);
    
    free(weights->rms_att_weight);
    free(weights->rms_ffn_weight);
    free(weights->rms_final_weight);
    
    // Free classifier (handling shared case)
    free(weights->wcls_weights);
    free(weights->wcls_scales);
}

int main() {
    std::string weights_path = get_data_path("models/weights.bin");
    std::string tokenizer_path = get_data_path("models/tokenizer.bin");
    
    Weights weights;
    read_checkpoint(weights_path, &weights);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab_size);
    
    const char *prompt = kDefaultPrompt;
    int safe_capacity = std::max((int)strlen(prompt) * 2 + 16, seq_len + 4);
    int *prompt_tokens = (int *)malloc(safe_capacity * sizeof(int));
    int num_prompt_tokens = 0;
    encode(&tokenizer, (char *)prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    
    if (num_prompt_tokens > safe_capacity) {
        std::cerr << "Warning: encode produced " << num_prompt_tokens 
                  << " tokens but buffer capacity was " << safe_capacity << "; clamping" << std::endl;
        num_prompt_tokens = safe_capacity;
    }
    
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    float *key_cache = (float *)calloc(n_layers * seq_len * kv_dim, sizeof(float));
    float *value_cache = (float *)calloc(n_layers * seq_len * kv_dim, sizeof(float));
    float *logits = (float *)calloc(vocab_size, sizeof(float));
    
    if (!key_cache || !value_cache || !logits || !prompt_tokens) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }
    
    int token = prompt_tokens[0];
    int pos = 0;
    int next = 0;
    int steps = kSteps;
    
    std::cout << "Input: " << prompt << std::endl;
    std::cout << "Output: ";
    
    for (int i = 0; i < steps; i++) {
        forward(
            weights.token_embedding_table,

            weights.wq_weights, weights.wq_scales,
            weights.wk_weights, weights.wk_scales,
            weights.wv_weights, weights.wv_scales,
            weights.wo_weights, weights.wo_scales,

            weights.w1_weights, weights.w1_scales,
            weights.w2_weights, weights.w2_scales,
            weights.w3_weights, weights.w3_scales,

            weights.rms_att_weight,
            weights.rms_ffn_weight,
            weights.rms_final_weight,

            weights.wcls_weights, weights.wcls_scales,

            key_cache, value_cache,

            logits,
            token, pos
        );
        
        softmax(logits, vocab_size);
        
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            float max_val = logits[0];
            int max_idx = 0;
            for (int j = 1; j < vocab_size; j++) {
                if (logits[j] > max_val) {
                    max_val = logits[j];
                    max_idx = j;
                }
            }
            next = max_idx;
        }
        
        if (next == 2) break;
        char *piece = decode(&tokenizer, token, next);
        std::cout << piece;
        std::cout.flush();
        
        token = next;
        pos++;
    }
    
    std::cout << std::endl;
    
    free_weights(&weights);
    free_tokenizer(&tokenizer);
    free(prompt_tokens);
    free(key_cache);
    free(value_cache);
    free(logits);
    
    return 0;
}