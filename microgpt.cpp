#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifdef DEBUG
#define LOG(msg) std::cerr << "[LOG] " << msg << std::endl
#else
#define LOG(msg)
#endif

using flt_t = float;
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
            auto x2pi = random() * 2.0 * std::numbers::pi;
            auto g2rad = std::sqrt(-2.0 * std::log(1.0 - random()));
            z = std::cos(x2pi) * g2rad;
            gauss_next = std::sin(x2pi) * g2rad;
        }

        return mu + z * sigma;
    }

    template<typename T, size_t N>
    std::vector<uint_t> choices(const std::array<T, N> &weights, const uint_t k = 1) {
        std::array<T, N> cum_weights{};
        std::partial_sum(weights.begin(), weights.end(), cum_weights.begin());

        auto total = cum_weights.back();
        assert(total > 0);

        std::vector<uint_t> res;
        res.reserve(k);
        for (uint_t i = 0; i < k; ++i) {
            res.push_back(std::distance(
                cum_weights.begin(),
                std::lower_bound(cum_weights.begin(), cum_weights.end(), random() * total)
            ));
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
constexpr uint_t SQRT_N_HEAD = 2;
static_assert(N_HEAD == SQRT_N_HEAD*SQRT_N_HEAD);
constexpr uint_t HEAD_DIM = N_EMBD / N_HEAD;
constexpr flt_t INV_SQRT_HEAD_DIM = flt_t{1} / SQRT_N_HEAD;
constexpr uint_t NO_CHILD = -1; // since children point to indices -> no child index = -1
constexpr uint_t MAX_VOCAB_SIZE = 27;
constexpr uint_t NUM_STEPS = 1000;


struct Value
{
    flt_t data{};
    const flt_t local_grad0{};
    const flt_t local_grad1{};
    const uint_t i_child0{};
    const uint_t i_child1{};
    const uint_t i_child2{};
};


class Arena : public std::vector<Value>
{
private:
    uint_t weights_end{0};

public:
    void weights_size_cutoff() { weights_end = size(); }
    void truncate() { resize(weights_end); }
    inline uint_t push_op(flt_t _data,
        uint_t _i_child0 = NO_CHILD, flt_t _local_grad0 = 0,
        uint_t _i_child1 = NO_CHILD, flt_t _local_grad1 = 0,
        uint_t _i_child2 = NO_CHILD
    ) {
        emplace_back(_data, _local_grad0, _local_grad1, _i_child0, _i_child1, _i_child2);
        return size() - 1;
    }
};

Arena arena{}; // memory management for all of our values
const auto &c_arena{arena}; // const view (sort of)

std::vector<flt_t> backward() {
    std::vector<flt_t> grad(c_arena.size(), flt_t{0});
    grad.back() = flt_t{1};

    auto v_it = c_arena.rbegin();
    for (auto g_it = grad.crbegin(); g_it != grad.crend(); ++v_it, ++g_it) {
        const auto g = *g_it;
        if (const auto i_c0 = v_it->i_child0; g != flt_t{0} && i_c0 != NO_CHILD) {
            grad[i_c0] += v_it->local_grad0 * g;
            if (const auto i_c1 = v_it->i_child1; i_c1 != NO_CHILD) {
                grad[i_c1] += v_it->local_grad1 * g;
                if (const auto i_c2 = v_it->i_child2; i_c2 != NO_CHILD) {
                    grad[i_c2] += g; // third child local_grad is always const
                }
            }
        }
    }

    return grad;
}


// operations (ternary)
inline uint_t vmul_add(const uint_t a, const uint_t b, const uint_t c) {
    return arena.push_op(c_arena[a].data * c_arena[b].data + c_arena[c].data,
        a, c_arena[b].data,
        b, c_arena[a].data,
        c
    );
}

// operations (binary)
inline uint_t vadd(const uint_t a, const uint_t b) { return arena.push_op(c_arena[a].data + c_arena[b].data, a, flt_t{1.0}, b, flt_t{1.0}); }
inline uint_t vsub(const uint_t a, const uint_t b) { return arena.push_op(c_arena[a].data - c_arena[b].data, a, flt_t{1.0}, b, flt_t{-1.0}); }
inline uint_t vmul(const uint_t a, const uint_t b) { return arena.push_op(c_arena[a].data * c_arena[b].data, a, c_arena[b].data, b, c_arena[a].data); }
inline uint_t vdiv(const uint_t a, const uint_t b) { return arena.push_op(c_arena[a].data / c_arena[b].data, a, flt_t{1.0}/c_arena[b].data, b, -c_arena[a].data /(c_arena[b].data * c_arena[b].data)); }

// operations (unary)
inline uint_t vneg(const uint_t a) { return arena.push_op(-c_arena[a].data, a, flt_t{-1.0}); }
inline uint_t vlog(const uint_t a) { return arena.push_op(std::log(c_arena[a].data), a, flt_t{1.0}/c_arena[a].data); }
inline uint_t vexp(const uint_t a) { const auto val = std::exp(c_arena[a].data); return arena.push_op(val, a, val); }
inline uint_t vrelu(const uint_t a) { return arena.push_op(std::max(flt_t{0.0}, c_arena[a].data), a, c_arena[a].data > flt_t{0}); }
inline uint_t vpow(const uint_t a, const flt_t n) { return arena.push_op(std::pow(c_arena[a].data, n), a, n * std::pow(c_arena[a].data, n - 1)); }
inline uint_t vinv_sqrt(const uint_t a) { const auto val = std::pow(c_arena[a].data + flt_t{1e-5}, flt_t{-0.5}); return arena.push_op(val, a, val / (flt_t{-2} * (c_arena[a].data + flt_t{1e-5}))); }

// operations with consts (1 node instead of 2)
inline uint_t add_const(const uint_t a, const flt_t c) { return arena.push_op(c_arena[a].data + c, a, flt_t{1.0}); }
inline uint_t sub_const(const uint_t a, const flt_t c) { return arena.push_op(c_arena[a].data - c, a, flt_t{1.0}); }
inline uint_t mul_const(const uint_t a, const flt_t c) { return arena.push_op(c_arena[a].data * c, a, c); }
inline uint_t div_const(const uint_t a, const flt_t c) { return arena.push_op(c_arena[a].data / c, a, flt_t{1.0} / c); }
inline uint_t vexp_sub_const(const uint_t a, const flt_t c) { const auto val = std::exp(c_arena[a].data - c); return arena.push_op(val, a, val); }


struct Matrix
{
    const uint_t data_start, rows, cols;

    Matrix(const uint_t rows, const uint_t cols, const flt_t std=0.08)
        : data_start(c_arena.size()) // start at the current arena pointer
        , rows(rows)
        , cols(cols)
    {
        for (uint_t i = 0; i < rows*cols; ++i) arena.push_op(rng.gauss(0,std));
    }

    uint_t at(const uint_t i, const uint_t j) const { return data_start + i * cols + j;}
};

struct Layer
{
    Matrix attn_wq, attn_wk, attn_wv, attn_wo;
    Matrix mlp_fc1, mlp_fc2;

    Layer(const uint_t n_embd)
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

    Model(const uint_t vocab_size, const uint_t n_embd, const uint_t block_size, const uint_t n_layer)
        : wte(vocab_size, n_embd)
        , wpe(block_size, n_embd)
        , lm_head(vocab_size, n_embd)
    {
        for(uint_t i = 0; i < n_layer; ++i) {
            layers.emplace_back(n_embd);
        }
    }
};

// [layers][timesteps][keys/values]
using KVCache = std::array<std::array<std::array<uint_t, N_EMBD>, BLOCK_SIZE>, N_LAYER>;

template<typename T, size_t N0, size_t N1>
void linear(std::array<T, N0> &out, const std::array<T, N1> &x, const Matrix &w) { // matrix * vector
    for(uint_t i = 0; i < w.rows; ++i) {
        const auto w_pos = w.at(i,0);
        auto sum = vmul(w_pos, x[0]);
        for(uint_t j = 1; j < w.cols; ++j) {
            sum = vmul_add(w_pos + j, x[j], sum);
        }
        out[i] = sum;
    }
}

template<typename T, size_t N>
void softmax(std::array<T, N> &out, const std::array<T, N> &logits, const uint_t logits_len) {
    flt_t max_val = std::numeric_limits<flt_t>::lowest();
    for (uint_t i = 0; i < logits_len; ++i) max_val = std::max(max_val, c_arena[logits[i]].data);

    std::array<uint_t, MAX_VOCAB_SIZE> exps{}; // indices of exps
    for (uint_t i = 0; i < logits_len; ++i) exps[i] = vexp_sub_const(logits[i], max_val);
    auto total = exps[0];
    for (uint_t i = 1; i < logits_len; ++i) total = vadd(total, exps[i]);
    for (uint_t i = 0; i < logits_len; ++i) out[i] = vdiv(exps[i], total);
}

template<typename T, size_t N>
void rmsnorm(std::array<T, N> &out, const std::array<T, N> &x, uint_t x_len) {
    auto total = vmul(x[0], x[0]);
    for (uint_t i = 1; i < x_len; ++i) total = vmul_add(x[i], x[i], total);
    total = div_const(total, x_len);
    auto scale = vinv_sqrt(total);
    for (uint_t i = 0; i < x_len; ++i) out[i] = vmul(x[i], scale);
}

template<typename T, size_t N>
void gpt(
    std::array<T, N> &logits_out,
    const uint_t token_id,
    const uint_t pos_id,
    KVCache &keys,
    KVCache &values,
    Model &state_dict
) {
    std::array<uint_t, N_EMBD> x{}; // joint token and position embedding
    std::array<uint_t, N_EMBD> tmp{}; // tmp array for rmsnorm, since we can't do it in place
    auto i_wte = state_dict.wte.at(token_id, 0);
    auto i_wpe = state_dict.wpe.at(pos_id, 0);
    for (uint_t j = 0; j < N_EMBD; ++j)
        x[j] = vadd(i_wte + j, i_wpe + j);
    rmsnorm(tmp, x, N_EMBD);
    x = tmp;

    for (uint_t i_layer = 0; i_layer < N_LAYER; ++i_layer) {
        auto &keys_layer = keys[i_layer];
        auto &vals_layer = values[i_layer];
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
        keys_layer[pos_id] = k;
        vals_layer[pos_id] = v;

        // multi-head attention
        std::array<uint_t, N_EMBD> x_attn{};
        auto num_timesteps = pos_id + 1;
        for(uint_t h = 0; h < N_HEAD; ++h) {
            auto hs = h * HEAD_DIM; // starting index of the full N_EMBD vector for head

            // computing attention dot(q_h, k_h[t]) / sqrt(head_dim)
            std::array<uint_t, BLOCK_SIZE> attention_logits{};
            for (uint_t t = 0; t < num_timesteps; ++t) {
                auto &keys_pos = keys_layer[t];
                auto sum = vmul(q[hs], keys_pos[hs]);
                for (uint_t j = 1;j < HEAD_DIM; ++j) {
                    sum = vmul_add(q[hs+j], keys_pos[hs + j], sum);
                }
                attention_logits[t] = mul_const(sum, INV_SQRT_HEAD_DIM);
            }
            // softmax
            std::array<uint_t, BLOCK_SIZE> attn_weights{};
            softmax(attn_weights, attention_logits, num_timesteps);

            // weighted sum of values
            for (uint_t j = 0; j < HEAD_DIM; ++j) {
                auto sum = vmul(attn_weights[0], vals_layer[0][hs + j]);
                for (uint_t t = 1; t < num_timesteps; ++t) {
                    sum = vmul_add(attn_weights[t], vals_layer[t][hs + j], sum);
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
    arena.reserve(
        MAX_VOCAB_SIZE * N_EMBD * 3 // accounting for wte, wpe, lm_head
        + N_LAYER * (4 * N_EMBD * N_EMBD + 2 * 4 * N_EMBD * N_EMBD) // accounting for attention heads and linear fc layers
    ); // it will grow automatically, this is just a hint to avoid a couple of realloc at the start

    if (!std::filesystem::exists("input.txt")) {
        std::cout << "Downloading input.txt ..." << std::endl;
        if (system("wget -q -O input.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt") != 0) {
            std::cerr << "Download failed" << std::endl;
            return -1;
        }
    }
    std::vector<std::string> docs;
    std::ifstream file("input.txt");
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) docs.push_back(line);
    }
    rng.shuffle(docs);
    std::cout << "We have " << docs.size() << " names." << std::endl;

    std::set<char> uchars{};
    for (auto& name: docs) uchars.insert(name.begin(), name.end());
    uint_t BOS = uchars.size(); // token id for a special Beginning of Sequence (BOS) token
    uint_t vocab_size = uchars.size() + 1;
    std::cout << "Vocab size is: " << vocab_size << std::endl;
    if (vocab_size > MAX_VOCAB_SIZE) [[unlikely]] {
        throw std::runtime_error("vocab_size (" + std::to_string(vocab_size) + ") exceeds MAX_VOCAB_SIZE (" + std::to_string(MAX_VOCAB_SIZE) + ")");
    }

    // build char lookup
    std::vector<char> idx_to_char(uchars.begin(), uchars.end());
    std::array<uint_t, 255> char_to_idx{}; // to cover all ASCII
    { uint_t idx = 0; for (char c : uchars) char_to_idx[uint_t(c)] = idx++; } // reverse idx_to_char to char_to_udx


    Model state_dict(vocab_size, N_EMBD, BLOCK_SIZE, N_LAYER);
    arena.weights_size_cutoff();
    const auto params_size = arena.size();
    std::cout << "Number of params: " << params_size << std::endl;

    const flt_t learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
    std::vector<flt_t> m(params_size, flt_t{0});
    std::vector<flt_t> v(params_size, flt_t{0});

    // training loop
    for (uint_t step = 0; step < NUM_STEPS; ++step) {
        // Take a document, tokenize it, surround it by BOS tokens
        const std::string doc = docs[step % docs.size()];
        std::array<uint_t, BLOCK_SIZE + 2> tokens{}; // context + 2 BOS
        uint_t token_len = 0;
        tokens[token_len++] = BOS;
        for (char ch:doc) { tokens[token_len++] = char_to_idx[uint_t(ch)]; }
        tokens[token_len++] = BOS;
        const auto n = std::min(BLOCK_SIZE, token_len - 1);

        //forward tokens through the model
        KVCache keys, values;
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
        const auto loss = div_const(total_losses, n);

        // backward pass
        const auto grad = backward();

        // adam optimizer
        const flt_t lr_t = learning_rate * (flt_t{1} - flt_t(step) / NUM_STEPS);
        const flt_t beta1_pow = flt_t{1} - std::pow(beta1,(step + 1));
        const flt_t beta2_pow = flt_t{1} - std::pow(beta2,(step + 1));
        for (uint_t i = 0; i < params_size; ++i) {
            const auto p_grad = grad[i]; // parameter gradient
            m[i] = beta1 * m[i] + (1 - beta1) * p_grad;
            v[i] = beta2 * v[i] + (1 - beta2) * p_grad * p_grad;
            arena[i].data -= lr_t * m[i] / ((std::sqrt(v[i] / beta2_pow) + eps_adam) * beta1_pow);
        }
        LOG("Step " << (step+1) << " / " << NUM_STEPS << " | loss " << c_arena[loss].data);
        LOG("Arena size: " << c_arena.size());
        arena.truncate(); // clean until end of weights values
    }

    const flt_t temp{0.5};
    const flt_t inv_temp{flt_t{1} / temp};
    std::cout << "\n\nTime for inference---------------" << std::endl;
    for (uint_t sample_idx = 0; sample_idx < 20; ++sample_idx) {
        KVCache keys, values;
        auto token_id = BOS;
        std::vector<char> samples;
        for (uint_t pos_id = 0; pos_id < BLOCK_SIZE; ++pos_id) {
            std::array<uint_t, MAX_VOCAB_SIZE> logits{};
            gpt(logits, token_id, pos_id, keys, values, state_dict);
            for (uint_t i = 0; i < vocab_size; ++i)
                logits[i] = mul_const(logits[i], inv_temp);

            std::array<uint_t, MAX_VOCAB_SIZE> probs{};
            softmax(probs, logits, vocab_size);

            std::array<flt_t, MAX_VOCAB_SIZE> weights{};
            for (uint_t i = 0; i < vocab_size; ++i) weights[i] = c_arena[probs[i]].data;
            token_id = rng.choices(weights)[0];
            if (token_id == BOS) [[unlikely]] break;
            samples.push_back(idx_to_char[token_id]);
        }
        std::string result(samples.begin(), samples.end());
        std::cout << "Sample: " << sample_idx << ": " << result << std::endl;
        arena.truncate(); // clean until end of weights values
    }
    return 0;
}