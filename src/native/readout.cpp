// Native first-position logits from Prism's ternary llama.cpp runtime.
// No sampling, generated answer, or truncated top-k distribution is involved.
// With an optional vision projector, requests may carry images that Prism's multimodal
// library (mtmd) encodes in place of media markers in the prompt.
// A request may also carry a shared prefix and several suffixes: the prefix is evaluated once,
// its sequence state is saved, and each suffix is read out from a restored copy of that state.
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
#include <list>
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

// Cross-request cache of prefix snapshots, most recently used first. Entries are matched on the
// exact prefix text and image strings, never on a hash, so a hit always means identical input.
static const size_t PREFIX_CACHE_ENTRIES = 4;
static const size_t PREFIX_CACHE_BYTES = size_t(2) << 30;
static const size_t MAX_SUFFIXES = 2048;

static std::vector<llama_token> tokenize_text(const llama_vocab *vocab, const std::string &s,
                                              bool add_special, bool parse_special) {
    int n = llama_tokenize(vocab, s.data(), s.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> tokens(std::abs(n));
    n = llama_tokenize(vocab, s.data(), s.size(), tokens.data(), tokens.size(), add_special, parse_special);
    if (n < 0) throw std::runtime_error("tokenization failed");
    tokens.resize(n);
    return tokens;
}

static std::vector<llama_token> label_ids(const llama_vocab *vocab, const json &labels) {
    std::vector<llama_token> ids;
    std::set<llama_token> unique;
    for (const auto &label : labels) {
        auto t = tokenize(vocab, label.get<std::string>(), false);
        if (t.size() != 1 || !unique.insert(t[0]).second)
            throw std::invalid_argument("labels must be distinct single tokens");
        ids.push_back(t[0]);
    }
    if (ids.empty() || ids.size() > 255) throw std::invalid_argument("invalid candidate count");
    return ids;
}

// Candidate logits of the last evaluated token and the log of the full-vocabulary normalizer.
static json candidates(llama_context *ctx, const std::vector<llama_token> &ids, int n_vocab) {
    const float *all = llama_get_logits_ith(ctx, -1);
    if (!all) throw std::runtime_error("no final logits");
    std::vector<double> logits;
    for (auto id : ids) {
        if (!std::isfinite(all[id])) throw std::runtime_error("non-finite candidate logit");
        logits.push_back(all[id]);
    }
    double peak = -INFINITY, sum = 0;
    for (int i = 0; i < n_vocab; ++i) peak = std::max<double>(peak, all[i]);
    for (int i = 0; i < n_vocab; ++i) sum += std::exp(all[i] - peak);
    return {{"logits", logits}, {"candidate_ids", ids}, {"log_normalizer", peak + std::log(sum)}};
}

// Decode text tokens on sequence 0 from position n_past, in batches like mtmd's text chunks,
// with logits for the last token only.
static void decode_text(llama_context *ctx, const std::vector<llama_token> &tokens, llama_pos n_past, int n_batch) {
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    try {
        for (size_t i = 0; i < tokens.size();) {
            batch.n_tokens = 0;
            for (; i < tokens.size() && batch.n_tokens < n_batch; ++i) {
                int j = batch.n_tokens++;
                batch.token[j] = tokens[i];
                batch.pos[j] = n_past++;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = false;
            }
            if (i == tokens.size()) batch.logits[batch.n_tokens - 1] = true;
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
        }
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
    llama_batch_free(batch);
}

// Decode and tokenize a prompt with images through mtmd, as the single-prompt path does.
static void media_chunks(mtmd_context *vision, const std::string &prompt, const json &images,
                         const std::string &marker, mtmd::input_chunks &chunks) {
    if (!vision) throw std::invalid_argument("image input requires the vision projector, which is not loaded");
    if (!images.is_array() || images.size() > MAX_IMAGES)
        throw std::invalid_argument("images must be a list of at most 8 base64 strings");
    if (count_markers(prompt, marker) != images.size())
        throw std::invalid_argument("the prompt must contain one media marker per image");
    mtmd::bitmaps bitmaps;
    for (size_t i = 0; i < images.size(); ++i) {
        auto bytes = base64_decode(images[i].get<std::string>());
        auto wrapped = mtmd_helper_bitmap_init_from_buf(vision, bytes.data(), bytes.size(), false);
        if (!wrapped.bitmap || mtmd_bitmap_is_audio(wrapped.bitmap)) {
            if (wrapped.bitmap) mtmd_bitmap_free(wrapped.bitmap);
            throw std::invalid_argument("image " + std::to_string(i) + " could not be decoded");
        }
        bitmaps.entries.emplace_back(wrapped.bitmap);
    }
    chunks.ptr.reset(mtmd_input_chunks_init());
    mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
    auto pointers = bitmaps.c_ptr();
    int32_t status = mtmd_tokenize(vision, chunks.ptr.get(), &text, pointers.data(), pointers.size());
    if (status == 1) throw std::invalid_argument("the prompt must contain one media marker per image");
    if (status == 2) throw std::invalid_argument("image preprocessing failed");
    if (status != 0) throw std::runtime_error("multimodal tokenization failed");
    if (chunks.size() == 0 || mtmd_input_chunk_get_type(chunks[chunks.size() - 1]) != MTMD_INPUT_CHUNK_TYPE_TEXT)
        throw std::invalid_argument("the prompt must end with text after the last image");
}

static size_t image_token_count(const mtmd::input_chunks &chunks) {
    size_t n = 0;
    for (size_t i = 0; i < chunks.size(); ++i)
        if (mtmd_input_chunk_get_type(chunks[i]) == MTMD_INPUT_CHUNK_TYPE_IMAGE)
            n += mtmd_input_chunk_get_n_tokens(chunks[i]);
    return n;
}

struct Snapshot {
    std::string key;
    std::vector<uint8_t> state;
    size_t prefix_tokens = 0, images = 0, image_tokens = 0;
    llama_pos n_past = 0;
    // Tokens of the prefix text after the last image (the whole prefix without images): the
    // part whose tokenization must not merge across the prefix boundary.
    std::string tail;
    std::vector<llama_token> tail_tokens;
};

struct Runtime {
    llama_context *ctx;
    const llama_vocab *vocab;
    mtmd_context *vision;
    std::string marker;
    int n_vocab;
    int n_batch;
    std::list<Snapshot> cache;

    size_t cache_bytes() const {
        size_t n = 0;
        for (const auto &e : cache) n += e.state.size() + e.key.size();
        return n;
    }
    json cache_info() const { return {{"entries", cache.size()}, {"host_bytes", cache_bytes()}}; }

    void restore(const Snapshot &s) {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_state_seq_set_data(ctx, s.state.data(), s.state.size(), 0) != s.state.size())
            throw std::runtime_error("restoring the prefix sequence state failed");
    }

    // Clear memory and evaluate a complete prompt (fallback for a suffix whose tokens differ).
    size_t prefill_full(const std::string &prompt, const json &images) {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (images.empty()) {
            auto tokens = tokenize(vocab, prompt, true);
            if (tokens.empty() || tokens.size() > llama_n_ctx(ctx))
                throw std::invalid_argument("prompt exceeds context or is empty; never truncated");
            for (size_t pos = 0; pos < tokens.size(); pos += n_batch) {
                auto count = std::min<size_t>(n_batch, tokens.size() - pos);
                if (llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, count)) != 0)
                    throw std::runtime_error("llama_decode failed");
            }
            return tokens.size();
        }
        mtmd::input_chunks chunks;
        media_chunks(vision, prompt, images, marker, chunks);
        size_t n = mtmd_helper_get_n_tokens(chunks.ptr.get());
        if (n > llama_n_ctx(ctx)) throw std::invalid_argument("prompt with images exceeds context; never truncated");
        llama_pos n_past = 0;
        if (mtmd_helper_eval_chunks(vision, ctx, chunks.ptr.get(), 0, 0, n_batch, true, &n_past) != 0)
            throw std::runtime_error("multimodal evaluation failed");
        return n;
    }

    json prefix_request(const json &request) {
        auto total_start = std::chrono::steady_clock::now();
        auto since = [](std::chrono::steady_clock::time_point t) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        };
        const std::string prefix = request.at("prefix");
        const json images = request.contains("images") ? request.at("images") : json::array();
        if (!images.is_array()) throw std::invalid_argument("images must be a list of at most 8 base64 strings");
        const auto &suffixes = request.at("suffixes");
        if (!suffixes.is_array() || suffixes.empty() || suffixes.size() > MAX_SUFFIXES)
            throw std::invalid_argument("suffixes must be a non-empty list of at most 2048 entries");
        const bool use_cache = !request.contains("cache") || request.at("cache").get<bool>();
        if (!images.empty() && !vision)
            throw std::invalid_argument("image input requires the vision projector, which is not loaded");
        if (count_markers(prefix, marker) != images.size())
            throw std::invalid_argument("the prefix must contain one media marker per image");

        std::string key;
        auto add_key = [&key](const std::string &part) { key += std::to_string(part.size()) + ":" + part; };
        add_key(prefix);
        for (const auto &image : images) add_key(image.get<std::string>());

        // Labels and suffix tokens first, so input errors stop the request before any evaluation.
        std::vector<std::string> texts;
        std::vector<std::vector<llama_token>> ids, suffix_tokens;
        for (const auto &suffix : suffixes) {
            texts.push_back(suffix.at("text").get<std::string>());
            if (count_markers(texts.back(), marker) != 0)
                throw std::invalid_argument("a suffix must not contain the media marker");
            ids.push_back(label_ids(vocab, suffix.at("labels")));
            suffix_tokens.push_back(tokenize_text(vocab, texts.back(), false, true));
            if (suffix_tokens.back().empty()) throw std::invalid_argument("a suffix must not be empty");
        }

        auto start = std::chrono::steady_clock::now();
        Snapshot *snapshot = nullptr;
        bool cache_hit = false, fresh = false;
        double prefix_ms = 0, snapshot_ms = 0, preprocess_ms = 0;
        Snapshot made;
        if (use_cache)
            for (auto it = cache.begin(); it != cache.end(); ++it)
                if (it->key == key) {
                    cache.splice(cache.begin(), cache, it);
                    snapshot = &cache.front();
                    cache_hit = true;
                    break;
                }
        if (!snapshot) {
            made.key = key;
            made.images = images.size();
            llama_memory_clear(llama_get_memory(ctx), true);
            if (images.empty()) {
                auto tokens = tokenize_text(vocab, prefix, true, true);
                if (tokens.empty() || tokens.size() > llama_n_ctx(ctx))
                    throw std::invalid_argument("prefix exceeds context or is empty; never truncated");
                decode_text(ctx, tokens, 0, n_batch);
                made.prefix_tokens = tokens.size();
                made.n_past = static_cast<llama_pos>(tokens.size());
                made.tail = prefix;
                made.tail_tokens = std::move(tokens);
            } else {
                mtmd::input_chunks chunks;
                media_chunks(vision, prefix, images, marker, chunks);
                made.image_tokens = image_token_count(chunks);
                made.prefix_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
                if (made.prefix_tokens > llama_n_ctx(ctx))
                    throw std::invalid_argument("prefix with images exceeds context; never truncated");
                made.tail = prefix.substr(prefix.rfind(marker) + marker.size());
                made.tail_tokens = tokenize_text(vocab, made.tail, false, true);
                // mtmd tokenizes the text after the last image on its own; the last chunk must end with it.
                size_t n_last = 0;
                const llama_token *last = mtmd_input_chunk_get_tokens_text(chunks[chunks.size() - 1], &n_last);
                if (n_last < made.tail_tokens.size() ||
                    !std::equal(made.tail_tokens.begin(), made.tail_tokens.end(), last + n_last - made.tail_tokens.size()))
                    made.tail_tokens.clear();  // Unexpected layout: every suffix falls back below.
                preprocess_ms = since(start);
                llama_pos n_past = 0;
                if (mtmd_helper_eval_chunks(vision, ctx, chunks.ptr.get(), 0, 0, n_batch, true, &n_past) != 0)
                    throw std::runtime_error("multimodal evaluation failed");
                made.n_past = n_past;
            }
            prefix_ms = since(start);
            auto snap_start = std::chrono::steady_clock::now();
            size_t size = llama_state_seq_get_size(ctx, 0);
            made.state.resize(size);
            if (size == 0 || llama_state_seq_get_data(ctx, made.state.data(), size, 0) != size)
                throw std::runtime_error("saving the prefix sequence state failed");
            snapshot_ms = since(snap_start);
            snapshot = &made;
            fresh = true;  // The memory still holds exactly the prefix.
        }

        json results = json::array();
        size_t fallbacks = 0;
        for (size_t k = 0; k < texts.size(); ++k) {
            auto suffix_start = std::chrono::steady_clock::now();
            size_t input_tokens = snapshot->prefix_tokens + suffix_tokens[k].size();
            if (input_tokens > llama_n_ctx(ctx))
                throw std::invalid_argument("prompt exceeds context; never truncated");
            // The prefix may be reused only where the full prompt's tokenization splits too.
            bool split = !snapshot->tail_tokens.empty();
            if (split) {
                auto whole = tokenize_text(vocab, snapshot->tail + texts[k], images.empty(), true);
                split = whole.size() == snapshot->tail_tokens.size() + suffix_tokens[k].size() &&
                        std::equal(snapshot->tail_tokens.begin(), snapshot->tail_tokens.end(), whole.begin()) &&
                        std::equal(suffix_tokens[k].begin(), suffix_tokens[k].end(),
                                   whole.begin() + snapshot->tail_tokens.size());
            }
            json result;
            double restore_ms = 0;
            if (split) {
                if (!fresh) {
                    auto restore_start = std::chrono::steady_clock::now();
                    restore(*snapshot);
                    restore_ms = since(restore_start);
                }
                decode_text(ctx, suffix_tokens[k], snapshot->n_past, n_batch);
                result = candidates(ctx, ids[k], n_vocab);
            } else {
                ++fallbacks;
                input_tokens = prefill_full(prefix + texts[k], images);
                result = candidates(ctx, ids[k], n_vocab);
                result["fallback"] = true;
            }
            fresh = false;
            result["input_tokens"] = input_tokens;
            result["suffix_tokens"] = suffix_tokens[k].size();
            result["restore_ms"] = restore_ms;
            result["prefill_ms"] = since(suffix_start);
            results.push_back(result);
        }

        json response = {{"results", results}, {"prefix_tokens", snapshot->prefix_tokens},
                         {"prefix_ms", prefix_ms}, {"snapshot_ms", snapshot_ms},
                         {"snapshot_bytes", snapshot->state.size()}, {"cache_hit", cache_hit},
                         {"fallbacks", fallbacks}};
        if (!images.empty()) {
            response["images"] = snapshot->images;
            response["image_tokens"] = snapshot->image_tokens;
            if (!cache_hit) response["image_preprocess_ms"] = preprocess_ms;
        }
        if (use_cache && !cache_hit && made.state.size() + key.size() <= PREFIX_CACHE_BYTES) {
            cache.push_front(std::move(made));
            while (cache.size() > PREFIX_CACHE_ENTRIES || cache_bytes() > PREFIX_CACHE_BYTES) cache.pop_back();
        }
        response["cache"] = cache_info();
        response["total_ms"] = since(total_start);
        return response;
    }
};

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: readout MODEL.gguf CONTEXT_TOKENS [MMPROJ.gguf]\n";
        return 2;
    }
#ifndef __APPLE__
    // Apple Silicon has exactly one Metal GPU and no CUDA device selection.
    const char *gpu = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!gpu || !std::regex_match(gpu, std::regex("GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"))) {
        std::cerr << "Set CUDA_VISIBLE_DEVICES to exactly one full GPU UUID\n";
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
                       {"image_min_tokens", IMAGE_MIN_TOKENS}, {"prefix_reuse", true},
                       {"prefix_cache_entries", PREFIX_CACHE_ENTRIES},
                       {"prefix_cache_bytes", PREFIX_CACHE_BYTES}}).dump() << std::endl;
    Runtime runtime{ctx, vocab, vision.get(), marker, n_vocab, static_cast<int>(cp.n_batch), {}};
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto request = json::parse(line);
            if (request.contains("prefix")) {
                std::cout << runtime.prefix_request(request).dump() << std::endl;
                continue;
            }
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
    runtime.cache.clear();
    vision.reset();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
