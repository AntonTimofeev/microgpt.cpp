#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifdef DEBUG
#define LOG(msg) std::cerr << "[LOG] " << msg << std::endl
#else
#define LOG(msg)
#endif

using flt_t = double;
using uint_t = unsigned;

typedef std::mersenne_twister_engine<
    uint32_t,
    32, 624, 397, 31,
    0x9908b0dfUL, 11,
    0xffffffffUL, 7,
    0x9d2c5680UL, 15,
    0xefc60000UL, 18, 1812433253UL> python_mt19937;

// Python's specific seeding algorithm
template<>
void python_mt19937::seed(uint32_t seed) {
    // > init_by_array
    // > init_genrand
    _M_x[0] = 19650218U;
    for (_M_p = 1; _M_p < state_size; ++_M_p) {
        _M_x[_M_p] = (1812433253U * (_M_x[_M_p-1] ^ (_M_x[_M_p-1] >> 30)) + _M_p);
    }
    // < init_genrand
    size_t i = 1;
    for (size_t k = state_size; k; --k) {
        _M_x[i] = (_M_x[i] ^ ((_M_x[i-1] ^ (_M_x[i-1] >> 30)) * 1664525U)) + seed;
        i++;
        if (i >= state_size) {
            _M_x[0] = _M_x[state_size-1];
            i = 1;
        }
    }
    for (size_t k = state_size-1; k; --k) {
        _M_x[i] = (_M_x[i] ^ ((_M_x[i-1] ^ (_M_x[i-1] >> 30)) * 1566083941U)) - i;
        i++;
        if (i >= state_size) {
            _M_x[0] = _M_x[state_size-1];
            i = 1;
        }
    }
    _M_x[0] = 0x80000000U;
    // < init_by_array
}

class PythonRandom
{
private:
    python_mt19937 rng;
    flt_t gauss_next;

public:
    PythonRandom(uint_t seed)
        : rng(seed)
        , gauss_next(std::numeric_limits<flt_t>::max())
    {}

    double random() {
        uint32_t a = rng() >> 5;  // Top 27 bits
        uint32_t b = rng() >> 6;  // Top 26 bits
        uint64_t value = (static_cast<uint64_t>(a) << 26) ^ b;
        return static_cast<double>(value) / (1ULL << 53);
    }

    uint_t getrandbits(const uint_t k) { return rng() >> (32 - k); }

    uint_t randbelow(const uint_t n) {
        if (n == 0) return 0;

        auto k = std::bit_width(n);
        auto r = getrandbits(k);
        while (r >= n) {
            r = getrandbits(k);
        }
        return r;
    }

    template<typename T>
    void shuffle(std::vector<T> &vec) {
        for (uint_t i = vec.size() - 1; i > 0; --i) {
            auto j = randbelow(i + 1);
            std::swap(vec[i], vec[j]);
        }
    }

    flt_t gauss(const flt_t mu, const flt_t sigma) {
        auto z = gauss_next;
        gauss_next = std::numeric_limits<flt_t>::max();

        if (z == std::numeric_limits<flt_t>::max()) {
            auto x2pi = random() * 2.0 * M_PI;
            auto g2rad = std::sqrt(-2.0 * std::log(1.0 - random()));
            z = std::cos(x2pi) * g2rad;
            gauss_next = std::sin(x2pi) * g2rad;
        }

        return mu + z * sigma;
    }
    template<typename T, size_t N>
    std::vector<uint_t> choices(const std::array<T, N> &weights, const uint_t k = 1) {
        std::array<T, N> cum_weights{weights};
        for (uint_t i = 1; i < N; ++i) {
            cum_weights[i] += cum_weights[i-1];
        }

        auto total = cum_weights.back();
        assert(total > 0);

        std::vector<uint_t> res;
        res.reserve(k);
        for (uint_t i = 0; i < k; ++i) {
            flt_t choice = random() * total;
            auto bin = std::lower_bound(cum_weights.begin(), cum_weights.end(), choice);
            res.push_back(std::distance(cum_weights.begin(), bin));
        }
        return res;
    }
};

PythonRandom rng(42); //random seed


// defining our model params
constexpr uint_t N_LAYER = 1;
constexpr uint_t N_EMBD = 16;
constexpr uint_t BLOCK_SIZE = 16;
constexpr uint_t N_HEAD = 4;
constexpr uint_t HEAD_DIM = N_EMBD / N_HEAD;
const      flt_t INV_SQRT_HEAD_DIM = 1.0 / std::sqrt(flt_t{HEAD_DIM});
constexpr uint_t NO_CHILD = -1; // since children point to indices -> no child index = -1
constexpr uint_t MAX_VOCAB_SIZE = 27;
constexpr uint_t NUM_STEPS = 1000;


struct Value
{
    flt_t data{};
    flt_t grad{};
    uint_t i_child0{};
    uint_t i_child1{};
    flt_t local_grad0{};
    flt_t local_grad1{};
};

class Arena : public std::vector<Value>
{
private:
    uint_t weights_end{0};

public:
    void init(uint_t n) { reserve(n); }
    void weights_size_cutoff() { weights_end = size(); }
    void truncate() { resize(weights_end); }
    void zero_grad() { std::for_each(begin(), begin() + weights_end, [](Value &v){ v.grad = 0; }); }

    inline uint_t push_op(flt_t d) { emplace_back(d, 0, NO_CHILD, NO_CHILD, 0, 0); return size() - 1; }
    inline uint_t push_op(flt_t d, uint_t i_c, flt_t g) { emplace_back(d, 0, i_c, NO_CHILD, g, 0); return size() - 1; }
    inline uint_t push_op(flt_t d, uint_t i_c0, flt_t g0, uint_t i_c1, flt_t g1) { emplace_back(d, 0, i_c0, i_c1, g0, g1); return size() - 1; }
};

Arena arena{};// memory management for all of our values

void backward() {
    arena.back().grad = 1;
    for (auto v_it = arena.rbegin(); v_it != arena.rend(); ++v_it) {
        auto g = v_it->grad;
        if (g == flt_t(0.0)) { continue; } // skip node when  grad is 0
        if (auto i_c0 = v_it->i_child0; i_c0 != NO_CHILD) {
            arena[i_c0].grad += v_it->local_grad0 * g;
            if (auto i_c1 = v_it->i_child1; i_c1 != NO_CHILD) {
                arena[i_c1].grad += v_it->local_grad1 * g;
            }
        }
    }
}

// operations (binary)
inline uint_t vadd(uint_t a, uint_t b) { return arena.push_op(arena[a].data + arena[b].data, a, 1.0, b, 1.0); }
inline uint_t vsub(uint_t a, uint_t b) { return arena.push_op(arena[a].data - arena[b].data, a, 1.0, b, -1.0); }
inline uint_t vmul(uint_t a, uint_t b) { return arena.push_op(arena[a].data * arena[b].data, a, arena[b].data, b, arena[a].data); }
inline uint_t vdiv(uint_t a, uint_t b) { return arena.push_op(arena[a].data / arena[b].data, a, 1.0/arena[b].data, b, -arena[a].data /(arena[b].data * arena[b].data)); }

// operations (unary)
inline uint_t vneg(uint_t a) { return arena.push_op(-arena[a].data, a, -1.0); }
inline uint_t vlog(uint_t a) { return arena.push_op(std::log(arena[a].data), a, 1.0 / arena[a].data); }
inline uint_t vexp(uint_t a) { auto e = std::exp(arena[a].data); return arena.push_op(e, a, e); }
inline uint_t vrelu(uint_t a) { return arena.push_op(std::max(flt_t{0.0}, arena[a].data), a, arena[a].data > 0); }
inline uint_t vpow(uint_t a, flt_t n) { return arena.push_op(std::pow(arena[a].data, n), a, n * std::pow(arena[a].data, n - 1)); }

// operations with consts (1 node instead of 2)
inline uint_t mul_const(uint_t a, flt_t c) { return arena.push_op(arena[a].data * c, a, c); }
inline uint_t div_const(uint_t a, flt_t c) { return arena.push_op(arena[a].data / c, a, 1.0 / c); }
inline uint_t add_const(uint_t a, flt_t c) { return arena.push_op(arena[a].data + c, a, 1.0); }
inline uint_t sub_const(uint_t a, flt_t c) { return arena.push_op(arena[a].data - c, a, 1.0); }


struct Matrix
{
    uint_t data_start;
    uint_t rows, cols;

    Matrix(uint_t rows, uint_t cols, flt_t std=0.08) : rows(rows), cols(cols) {
        data_start = arena.size(); // start at the current arena pointer
        for (uint_t i = 0; i < rows*cols; ++i) arena.push_op(rng.gauss(0,std));
    }

    uint_t at(uint_t i, uint_t j) const { return data_start + i * cols + j;}
};

struct Layer
{
    Matrix attn_wq, attn_wk, attn_wv, attn_wo;
    Matrix mlp_fc1, mlp_fc2;

    Layer(uint_t n_embd)
        : attn_wq(n_embd, n_embd)
        , attn_wk(n_embd, n_embd)
        , attn_wv(n_embd, n_embd)
        , attn_wo(n_embd, n_embd)
        , mlp_fc1(4 * n_embd, n_embd)
        , mlp_fc2(n_embd, 4 * n_embd)
    {}
};

struct Model
{
    Matrix wte, wpe, lm_head;
    std::vector<Layer> layers;

    Model(uint_t vocab_size, uint_t n_embd, uint_t block_size, uint_t n_layer)
        : wte(vocab_size, n_embd)
        , wpe(block_size, n_embd)
        , lm_head(vocab_size, n_embd)
    {
        for(uint_t i = 0; i < n_layer; ++i) {
            layers.emplace_back(n_embd);
        }
    }

    std::vector<uint_t> params() {
        std::vector<uint_t> p;
        p.reserve(4192);
        // helper to add all elements of a matrix
        auto add = [&](Matrix& m) {
            for (uint_t i = 0; i < m.rows*m.cols; ++i) p.push_back(m.data_start + i); // add indices to matrix values in arena
        };
        add(wte);
        add(wpe);
        add(lm_head);
        for (auto& layer : layers) {
            add(layer.attn_wq);
            add(layer.attn_wk);
            add(layer.attn_wv);
            add(layer.attn_wo);
            add(layer.mlp_fc1);
            add(layer.mlp_fc2);
        }
        return p;
    }
};

/**
 *
 * keys[layer][timestep][dimension]
 *        |        |         |
 *        |        |         -- which of the 16 numbers in the key vector (0..N_EMBD-1)
 *        |        -- which token position was processed (grows: 0, 1, 2, ...)
 *        -- which transformer layer (0..N_LAYER-1). Each layer has its own Q/K/V weights,
 *           so each layer produces different keys and values
 *
 * key = [k0, k1, k2, k3,   k4, k5, k6, k7,   k8, k9, k10, k11,   k12, k13, k14, k15]
 *        --- head 0 -----  ---- head 1 ----  ---- head 2 -----   ----- head 3 -----
 **/
struct FlatKVCache
{
    std::vector<uint_t> data; // indices to find the values we need
    uint_t n_layer, dim;
    std::array<uint_t, N_LAYER> counts{}; // timesteps per layer (max N_LAYER layers)

    FlatKVCache(uint_t n_layer, uint_t d)
        : n_layer(n_layer)
        , dim(d)
    {
        data.reserve(n_layer * BLOCK_SIZE * dim); // pre-alloc for up to BLOCK_SIZE timesteps (context length)
    }

    void push(uint_t i_layer, const uint_t *vals) {
        uint_t base = 0;
        for (uint_t l = 0; l < i_layer; l++) base += counts[l] * dim; // skip past all time steps for all previous layers
        base += counts[i_layer] * dim; // skip past this existing time steps for this layer
        // insert at the right position
        data.insert(data.begin() + base, vals, vals + dim);
        counts[i_layer]++; // we are now at the next time step for layer: i_layer
    }

    uint_t get(uint_t i_layer, uint_t t, uint_t d) {
        uint_t base = 0;
        for (uint_t l = 0; l < i_layer; l++) base += counts[l] * dim;
        return data[base + t * dim + d];
    }

    uint_t num_timesteps(uint_t i_layer) const { return counts[i_layer]; }
};

template<typename T, size_t N0, size_t N1>
void linear(std::array<T, N0> &out, const std::array<T, N1> &x, Matrix &w) { // matrix * vector
    for(uint_t i = 0; i < w.rows; ++i) {
        auto sum = vmul(w.at(i,0), x[0]);
        for(uint_t j = 1; j < w.cols; ++j) {
            sum = vadd(sum, vmul(w.at(i,j), x[j]));
        }
        out[i] = sum;
    }
}

template<typename T, size_t N>
void softmax(std::array<T, N> &out, const std::array<T, N> &logits, uint_t logits_len) {
    auto max_val = arena[logits[0]].data;
    for (uint_t i = 0; i < logits_len; ++i) max_val = std::max(arena[logits[i]].data, max_val);
    std::array<uint_t, MAX_VOCAB_SIZE> exps{}; // indices of exps
    for (uint_t i = 0; i < logits_len; ++i) exps[i] = vexp(sub_const(logits[i], max_val));
    auto total = exps[0];
    for (uint_t i = 1; i < logits_len; ++i) total = vadd(total, exps[i]);
    for (uint_t i = 0; i < logits_len; ++i) out[i] = vdiv(exps[i], total);
}

template<typename T, size_t N>
void rmsnorm(std::array<T, N> &out, const std::array<T, N> &x, uint_t x_len) {
    auto total = vmul(x[0], x[0]);
    for (uint_t i = 1; i < x_len; ++i) total = vadd(total, vmul(x[i], x[i]));
    total = div_const(total, x_len);
    auto scale = vpow(add_const(total,flt_t{1e-5}), flt_t{-0.5});
    for (uint_t i = 0; i < x_len; ++i) out[i] = vmul(x[i], scale);
}

template<typename T, size_t N>
void gpt(
    std::array<T, N> &logits_out,
    const uint_t token_id,
    const uint_t pos_id,
    FlatKVCache &keys,
    FlatKVCache &values,
    Model &state_dict
) {
    std::array<uint_t, N_EMBD> x{}; // joint token and position embedding
    std::array<uint_t, N_EMBD> tmp{}; // tmp array for rmsnorm, since we can't do it in place
    for (uint_t j = 0; j < N_EMBD; ++j)
        x[j] = vadd(state_dict.wte.at(token_id, j), state_dict.wpe.at(pos_id, j));
    rmsnorm(tmp, x, N_EMBD);
    x = tmp;

    for (uint_t i_layer=0; i_layer < N_LAYER; ++i_layer) {
        // save residual
        std::array<uint_t, N_EMBD> x_residual{x};
        // rmsnorm
        rmsnorm(tmp, x, N_EMBD);
        x = tmp;
        // Q, K, V
        std::array<uint_t, N_EMBD> q, k, v;
        linear(q, x, state_dict.layers[i_layer].attn_wq);
        linear(k, x, state_dict.layers[i_layer].attn_wk);
        linear(v, x, state_dict.layers[i_layer].attn_wv);
        keys.push(i_layer, k.data());
        values.push(i_layer, v.data());
        // multi-head attention
        std::array<uint_t, N_EMBD> x_attn{};
        auto num_timesteps = keys.num_timesteps(i_layer);
        for(uint_t h = 0; h < N_HEAD; ++h) {
            auto hs = h * HEAD_DIM; // starting index of the full N_EMBD vector for head

            // computing attention dot(q_h, k_h[t]) / sqrt(head_dim)
            std::array<uint_t, BLOCK_SIZE> attention_logits{};
            for (uint_t t = 0; t < num_timesteps; ++t) {
                auto sum = vmul(q[hs],keys.get(i_layer, t, hs));
                for (uint_t j = 1;j < HEAD_DIM; ++j) {
                    sum = vadd(sum, vmul(q[hs+j],keys.get(i_layer, t, hs + j)));
                }
                attention_logits[t] = mul_const(sum, INV_SQRT_HEAD_DIM);
            }
            // softmax
            std::array<uint_t, BLOCK_SIZE> attn_weights{};
            softmax(attn_weights, attention_logits, num_timesteps);

            // weighted sum of values
            for (uint_t j = 0; j < HEAD_DIM; ++j) {
                auto sum = vmul(attn_weights[0], values.get(i_layer, 0, hs+ j));
                for (uint_t t = 1; t < num_timesteps; ++t) {
                    sum = vadd(sum,vmul(attn_weights[t],values.get(i_layer, t, hs + j)));
                }
                x_attn[hs + j] = sum;
            }
        }

        // output projection
        linear(x, x_attn, state_dict.layers[i_layer].attn_wo);

        // residual connection
        for (uint_t i = 0; i < N_EMBD; ++i) {
            x[i] = vadd(x[i], x_residual[i]);
        }
        // MLP block
        x_residual = x;
        rmsnorm(tmp, x, N_EMBD);
        x = tmp;

        std::array<uint_t, 4 * N_EMBD> mlp_hidden{}; // since shape of mlp_fc1 is (4*N_EMBD, N_EMBD)
        linear(mlp_hidden, x, state_dict.layers[i_layer].mlp_fc1);
        for (uint_t i = 0; i < 4 * N_EMBD; ++i) mlp_hidden[i] = vrelu(mlp_hidden[i]);
        linear(x, mlp_hidden,  state_dict.layers[i_layer].mlp_fc2);
        for (uint_t i = 0; i < N_EMBD; ++i) x[i] = vadd(x[i], x_residual[i]);
    }
    linear(logits_out, x, state_dict.lm_head);
}

int main() {
    arena.init(
        MAX_VOCAB_SIZE * N_EMBD * 3 // accounting for wte, wpe, lm_head
        + N_LAYER * (4 * N_EMBD * N_EMBD + 2 * 4 * N_EMBD * N_EMBD) // accounting for attention heads and linear fc layers
    ); // it will grow automatically, this is just a hint to avoid a couple of realloc at the start

    if (!std::filesystem::exists("input.txt")) {
        LOG("Downloading input.txt ...");
         if (system("wget -q -O input.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt") != 0) LOG("Download failed");
    }
    std::vector<std::string> docs;
    std::ifstream file("input.txt");
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) docs.push_back(line);
    }
    rng.shuffle(docs);
    LOG("We have "<<docs.size()<<" names.");

    std::set<char> uchars{};
    for (auto& name: docs) uchars.insert(name.begin(), name.end());
    auto BOS = uchars.size(); // token id for a special Beginning of Sequence (BOS) token
    auto vocab_size = uchars.size() + 1;
    LOG("Vocab size is: "<<vocab_size);
    if (vocab_size > MAX_VOCAB_SIZE) {
        throw std::runtime_error("vocab_size (" + std::to_string(vocab_size) + ") exceeds MAX_VOCAB_SIZE (" + std::to_string(MAX_VOCAB_SIZE) + ")");
    }

    // build char lookup
    std::vector<char> idx_to_char(uchars.begin(), uchars.end());
    std::array<uint_t, 255> char_to_idx{}; // to cover all ASCII
    { uint_t idx = 0; for (char c : uchars) char_to_idx[uint_t(c)] = idx++; } // reverse idx_to_char to char_to_udx


    Model state_dict(vocab_size, N_EMBD, BLOCK_SIZE, N_LAYER);
    arena.weights_size_cutoff();
    auto params = state_dict.params();
    LOG("Number of params: "<<params.size());

    flt_t learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
    std::vector<flt_t> m(params.size(), 0.0);
    std::vector<flt_t> v(params.size(), 0.0);

    // training loop
    for (uint_t step = 0; step < NUM_STEPS; ++step) {
        // Take a document, tokenize it, surround it by BOS tokens
        std::string doc = docs[step%docs.size()];
        std::array<uint_t, BLOCK_SIZE + 2> tokens{}; // context + 2 BOS
        uint_t token_len = 0;
        tokens[token_len++] = BOS;
        for (char ch:doc) { tokens[token_len++] = char_to_idx[uint_t(ch)]; }
        tokens[token_len++] = BOS;
        auto n = std::min(BLOCK_SIZE, token_len - 1);

        //forward tokens through the model
        FlatKVCache keys(N_LAYER, N_EMBD), values(N_LAYER, N_EMBD);
        std::array<uint_t, BLOCK_SIZE> losses{};
        uint_t n_losses = 0;
        for (uint_t pos_id = 0; pos_id < n; ++pos_id) {
            auto token_id = tokens[pos_id];
            auto target_id = tokens[pos_id+1];
            std::array<uint_t, MAX_VOCAB_SIZE> logits{};
            gpt(logits, token_id, pos_id, keys, values, state_dict);
            std::array<uint_t, MAX_VOCAB_SIZE> probs{};
            softmax(probs, logits, vocab_size);
            losses[n_losses++] = vneg(vlog(probs[target_id]));
        }

        auto total_losses = losses[0];
        for (uint_t i = 1; i < n_losses; ++i) total_losses = vadd(total_losses, losses[i]);
        auto loss = mul_const(total_losses, 1.0/n);

        // backward pass
        backward();

        // adam optimizer
        flt_t lr_t = learning_rate*(1-(flt_t)step/NUM_STEPS);
        flt_t beta1_pow = std::pow(beta1,(step + 1));
        flt_t beta2_pow = std::pow(beta2,(step + 1));
        for (uint_t i = 0; i < params.size(); ++i) {
            auto i_p = params[i]; // parameter index
            auto p_grad = arena[i_p].grad; // parameter gradient
            m[i] = beta1 * m[i] + (1 - beta1) * p_grad;
            v[i] = beta2 * v[i] + (1 - beta2) * p_grad * p_grad;
            auto m_hat = m[i] / (1 - beta1_pow);
            auto v_hat = v[i] / (1 - beta2_pow);
            arena[i_p].data -= lr_t*m_hat / (std::sqrt(v_hat) + eps_adam);
        }
        LOG("Step " << (step+1) << " / " << NUM_STEPS << " | loss " << arena[loss].data);
        LOG("Arena size: " << arena.size());
        arena.truncate(); // clean until end of weights values
        arena.zero_grad();
    }

    flt_t temperature = 0.5;
    std::cout << "\n\nTime for inference---------------" << std::endl;
    for (uint_t sample_idx = 0; sample_idx < 20; ++sample_idx) {
        FlatKVCache keys(N_LAYER, N_EMBD), values(N_LAYER, N_EMBD);
        auto token_id = BOS;
        std::vector<char> samples;
        for (uint_t pos_id = 0; pos_id < BLOCK_SIZE; ++pos_id) {
            std::array<uint_t, MAX_VOCAB_SIZE> logits{};
            gpt(logits, token_id, pos_id, keys, values, state_dict);
            for (uint_t i = 0; i < vocab_size; ++i)
                logits[i] = mul_const(logits[i],1.0/temperature);
            std::array<uint_t, MAX_VOCAB_SIZE> probs{};
            softmax(probs, logits, vocab_size);

            std::array<flt_t, MAX_VOCAB_SIZE> weights{};
            for (uint_t i = 0; i < vocab_size; ++i) weights[i] = arena[probs[i]].data;
            token_id = rng.choices(weights)[0];
            if (token_id == BOS) break;
            samples.push_back(idx_to_char[token_id]);
        }
        std::string result(samples.begin(), samples.end());
        std::cout << "Sample: " << sample_idx << ": " << result << std::endl;
        arena.truncate(); // clean until end of weights values
    }
    return 0;
}