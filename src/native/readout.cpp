// Native first-position logits from Prism's ternary llama.cpp runtime.
// No sampling, generated answer, or truncated top-k distribution is involved.
// With an optional vision projector, requests may carry images that Prism's multimodal
// library (mtmd) encodes in place of media markers in the prompt.
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &s,
                                        bool special) {
    int n = llama_tokenize(vocab, s.data(), s.size(), nullptr, 0, special, special);
    std::vector<llama_token> tokens(std::abs(n));
    n = llama_tokenize(vocab, s.data(), s.size(), tokens.data(), tokens.size(), special, special);
    if (n < 0) throw std::runtime_error("tokenization failed");
    tokens.resize(n);
    return tokens;
}

// Images arrive as standard base64 (with padding); anything else is an input error.
static std::vector<unsigned char> base64_decode(const std::string &s) {
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (s.empty() || s.size() % 4 != 0) throw std::invalid_argument("image is not valid base64");
    std::vector<unsigned char> out;
    out.reserve(s.size() / 4 * 3);
    for (size_t i = 0; i < s.size(); i += 4) {
        int v[4], pad = 0;
        for (int j = 0; j < 4; ++j) {
            char c = s[i + j];
            if (c == '=' && i + 4 == s.size() && j >= 2) { v[j] = 0; ++pad; continue; }
            auto k = alphabet.find(c);
            if (k == std::string::npos || pad) throw std::invalid_argument("image is not valid base64");
            v[j] = static_cast<int>(k);
        }
        unsigned n = (v[0] << 18) | (v[1] << 12) | (v[2] << 6) | v[3];
        out.push_back((n >> 16) & 0xff);
        if (pad < 2) out.push_back((n >> 8) & 0xff);
        if (pad < 1) out.push_back(n & 0xff);
    }
    return out;
}

static size_t count_markers(const std::string &prompt, const std::string &marker) {
    size_t count = 0;
    for (auto pos = prompt.find(marker); pos != std::string::npos; pos = prompt.find(marker, pos + marker.size()))
        ++count;
    return count;
}

static const size_t MAX_IMAGES = 8;
static const int IMAGE_MIN_TOKENS = 1024;  // Matches the production Bonsai server (--image-min-tokens 1024).

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: readout MODEL.gguf CONTEXT_TOKENS [MMPROJ.gguf]\n";
        return 2;
    }
#ifndef __APPLE__
    // Apple Silicon has exactly one Metal GPU and no CUDA device selection.
    const char *vendor = std::getenv("SHINGI_GPU_VENDOR");
    const bool rocm = vendor && std::string(vendor) == "rocm";
    const char *gpu = std::getenv(rocm ? "ROCR_VISIBLE_DEVICES" : "CUDA_VISIBLE_DEVICES");
    if (!gpu || !(rocm ? std::regex_match(gpu, std::regex("[0-9]+"))
                       : std::regex_match(gpu, std::regex("GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")))) {
        std::cerr << (rocm ? "Set ROCR_VISIBLE_DEVICES to exactly one GPU index"
                           : "Set CUDA_VISIBLE_DEVICES to exactly one full GPU UUID") << "\n";
        return 2;
    }
#endif
    int context = std::stoi(argv[2]);
    if (context < 512 || context > 16384) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    if (!llama_supports_gpu_offload()) return 2;
    size_t gpu_count = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        // Unified-memory GPUs such as the DGX Spark GB10 register as integrated GPUs;
        // Apple Metal (MTL0) registers as a GPU.
        if (auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
            type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) ++gpu_count;
    if (gpu_count != 1) {
        std::cerr << "Expected one visible GPU; refusing CPU fallback or multiple devices\n";
        return 2;
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 12;
    cp.n_threads_batch = 12;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 1; }
    const auto *vocab = llama_model_get_vocab(model);
    mtmd::context_ptr vision;
    if (argc == 4) {
        auto vp = mtmd_context_params_default();
        vp.use_gpu = true;
        vp.print_timings = false;
        vp.n_threads = cp.n_threads;
        vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        vp.image_min_tokens = IMAGE_MIN_TOKENS;
        vision.reset(mtmd_init_from_file(argv[3], model, vp));
        if (!vision || !mtmd_support_vision(vision.get())) {
            std::cerr << "failed to load the vision projector " << argv[3] << "\n";
            vision.reset();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }
    const std::string marker = mtmd_default_marker();
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::cout << json({{"ready", true}, {"context_tokens", llama_n_ctx(ctx)},
                       {"vocab_size", n_vocab}, {"vision", static_cast<bool>(vision)},
                       {"media_marker", marker}, {"max_images", MAX_IMAGES},
                       {"image_min_tokens", IMAGE_MIN_TOKENS}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto request = json::parse(line);
            std::string prompt = request.at("prompt");
            const bool has_images = request.contains("images") && !request.at("images").empty();
            std::vector<llama_token> tokens;
            if (!has_images) {
                tokens = tokenize(vocab, prompt, true);
                if (tokens.empty() || tokens.size() > llama_n_ctx(ctx))
                    throw std::invalid_argument("prompt exceeds context or is empty; never truncated");
            }
            std::vector<llama_token> ids;
            std::set<llama_token> unique;
            for (const auto &label : request.at("labels")) {
                auto t = tokenize(vocab, label.get<std::string>(), false);
                if (t.size() != 1 || !unique.insert(t[0]).second)
                    throw std::invalid_argument("labels must be distinct single tokens");
                ids.push_back(t[0]);
            }
            if (ids.empty() || ids.size() > 255) throw std::invalid_argument("invalid candidate count");
            json image_info;
            size_t input_tokens = tokens.size();
            llama_memory_clear(llama_get_memory(ctx), true);
            auto start = std::chrono::steady_clock::now();
            if (!has_images) {
                for (size_t pos = 0; pos < tokens.size(); pos += cp.n_batch) {
                    auto count = std::min<size_t>(cp.n_batch, tokens.size() - pos);
                    auto batch = llama_batch_get_one(tokens.data() + pos, count);
                    if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
                }
            } else {
                if (!vision) throw std::invalid_argument("image input requires the vision projector, which is not loaded");
                const auto &images = request.at("images");
                if (!images.is_array() || images.size() > MAX_IMAGES)
                    throw std::invalid_argument("images must be a list of at most 8 base64 strings");
                if (count_markers(prompt, marker) != images.size())
                    throw std::invalid_argument("the prompt must contain one media marker per image");
                mtmd::bitmaps bitmaps;
                for (size_t i = 0; i < images.size(); ++i) {
                    auto bytes = base64_decode(images[i].get<std::string>());
                    auto wrapped = mtmd_helper_bitmap_init_from_buf(vision.get(), bytes.data(), bytes.size(), false);
                    if (!wrapped.bitmap || mtmd_bitmap_is_audio(wrapped.bitmap)) {
                        if (wrapped.bitmap) mtmd_bitmap_free(wrapped.bitmap);
                        throw std::invalid_argument("image " + std::to_string(i) + " could not be decoded");
                    }
                    bitmaps.entries.emplace_back(wrapped.bitmap);
                }
                mtmd::input_chunks chunks(mtmd_input_chunks_init());
                mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
                auto pointers = bitmaps.c_ptr();
                int32_t status = mtmd_tokenize(vision.get(), chunks.ptr.get(), &text, pointers.data(), pointers.size());
                if (status == 1) throw std::invalid_argument("the prompt must contain one media marker per image");
                if (status == 2) throw std::invalid_argument("image preprocessing failed");
                if (status != 0) throw std::runtime_error("multimodal tokenization failed");
                size_t image_tokens = 0;
                for (size_t i = 0; i < chunks.size(); ++i)
                    if (mtmd_input_chunk_get_type(chunks[i]) == MTMD_INPUT_CHUNK_TYPE_IMAGE)
                        image_tokens += mtmd_input_chunk_get_n_tokens(chunks[i]);
                input_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
                if (input_tokens > llama_n_ctx(ctx))
                    throw std::invalid_argument("prompt with images exceeds context; never truncated");
                if (chunks.size() == 0 || mtmd_input_chunk_get_type(chunks[chunks.size() - 1]) != MTMD_INPUT_CHUNK_TYPE_TEXT)
                    throw std::invalid_argument("the prompt must end with text after the last image");
                double preprocess_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                llama_pos n_past = 0;
                if (mtmd_helper_eval_chunks(vision.get(), ctx, chunks.ptr.get(), 0, 0, cp.n_batch, true, &n_past) != 0)
                    throw std::runtime_error("multimodal evaluation failed");
                image_info = {{"images", images.size()}, {"image_tokens", image_tokens},
                              {"image_preprocess_ms", preprocess_ms}};
            }
            const float *all = llama_get_logits_ith(ctx, -1);
            if (!all) throw std::runtime_error("no final logits");
            std::vector<double> logits;
            for (auto id : ids) {
                if (!std::isfinite(all[id])) throw std::runtime_error("non-finite candidate logit");
                logits.push_back(all[id]);
            }
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            // Log of the full-vocabulary normalizer, so callers can report the answer labels' total mass.
            double peak = -INFINITY, sum = 0;
            for (int i = 0; i < n_vocab; ++i) peak = std::max<double>(peak, all[i]);
            for (int i = 0; i < n_vocab; ++i) sum += std::exp(all[i] - peak);
            json response = {{"logits", logits}, {"candidate_ids", ids}, {"input_tokens", input_tokens},
                             {"prefill_ms", ms}, {"log_normalizer", peak + std::log(sum)}};
            if (has_images) response.update(image_info);
            std::cout << response.dump() << std::endl;
        } catch (const std::invalid_argument &e) {
            std::cout << json({{"error", e.what()}, {"error_kind", "input"}}).dump() << std::endl;
        } catch (const json::exception &e) {
            std::cout << json({{"error", e.what()}, {"error_kind", "input"}}).dump() << std::endl;
        } catch (const std::exception &e) {
            std::cout << json({{"error", e.what()}, {"error_kind", "runtime"}}).dump() << std::endl;
        }
    }
    vision.reset();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
