#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

class Fib2584Board {
public:
    using raw_t = unsigned __int128;

    static constexpr int kSize = 4;
    static constexpr int kCells = 16;
    static constexpr int kCellBits = 5;
    static constexpr int kRowBits = 20;
    static constexpr int kRowMask = (1 << kRowBits) - 1;
    static constexpr int kCellMask = (1 << kCellBits) - 1;

    Fib2584Board() : raw_(0) {}

    void clear() { raw_ = 0; }
    raw_t raw() const { return raw_; }
    void set_raw(raw_t raw) { raw_ = raw; }

    int at_code(int idx) const {
        return int((raw_ >> (idx * kCellBits)) & raw_t(kCellMask));
    }

    void set_code(int idx, int code) {
        const raw_t mask = raw_t(kCellMask) << (idx * kCellBits);
        raw_ = (raw_ & ~mask) | (raw_t(code & kCellMask) << (idx * kCellBits));
    }

    int fetch_row(int row) const {
        return int((raw_ >> (row * kRowBits)) & raw_t(kRowMask));
    }

    void place_row(int row, int packed) {
        const raw_t mask = raw_t(kRowMask) << (row * kRowBits);
        raw_ = (raw_ & ~mask) | (raw_t(packed & kRowMask) << (row * kRowBits));
    }

    static uint64_t tile_value_from_code(int code) {
        static const std::array<uint64_t, 32> fib = []() {
            std::array<uint64_t, 32> v{};
            v[0] = 0;
            v[1] = 1;
            v[2] = 2;
            for (size_t i = 3; i < v.size(); ++i) v[i] = v[i - 1] + v[i - 2];
            return v;
        }();
        return fib[std::min<int>(code, int(fib.size()) - 1)];
    }

    static int code_from_tile_value(uint64_t value) {
        if (value == 0) return 0;
        for (int code = 1; code < 32; ++code) {
            if (tile_value_from_code(code) == value) return code;
        }
        throw std::invalid_argument("invalid Fib-2584 tile value in state");
    }

    int tile_value(int idx) const {
        return int(tile_value_from_code(at_code(idx)));
    }

    int max_code() const {
        int best = 0;
        for (int i = 0; i < 16; ++i) best = std::max(best, at_code(i));
        return best;
    }

    int max_tile_value() const {
        return int(tile_value_from_code(max_code()));
    }

    int empty_count() const {
        int count = 0;
        for (int i = 0; i < 16; ++i) count += (at_code(i) == 0);
        return count;
    }

    std::array<int, 16> to_values_flat() const {
        std::array<int, 16> values{};
        for (int i = 0; i < 16; ++i) values[i] = tile_value(i);
        return values;
    }

    std::array<uint8_t, 16> to_codes_flat() const {
        std::array<uint8_t, 16> codes{};
        for (int i = 0; i < 16; ++i) codes[i] = static_cast<uint8_t>(at_code(i));
        return codes;
    }

    void init(std::mt19937& rng) {
        clear();
        popup(rng);
        popup(rng);
    }

    bool popup(std::mt19937& rng) {
        std::array<int, 16> empty{};
        int n = 0;
        for (int i = 0; i < 16; ++i) {
            if (at_code(i) == 0) empty[n++] = i;
        }
        if (n == 0) return false;

        std::uniform_int_distribution<int> pick_pos(0, n - 1);
        std::uniform_real_distribution<double> pick_prob(0.0, 1.0);

        const int code = (pick_prob(rng) < 0.9) ? 1 : 2;
        set_code(empty[pick_pos(rng)], code);
        return true;
    }

    int move(int action) {
        switch (action) {
            case 0: return move_up();
            case 1: return move_down();
            case 2: return move_left();
            case 3: return move_right();
            default: throw std::invalid_argument("action must be one of {0,1,2,3}");
        }
    }

    bool is_move_legal(int action) const {
        Fib2584Board copy = *this;
        return copy.move(action) >= 0;
    }

    std::vector<int> legal_actions() const {
        std::vector<int> acts;
        acts.reserve(4);
        for (int a = 0; a < 4; ++a) {
            if (is_move_legal(a)) acts.push_back(a);
        }
        return acts;
    }

    bool has_legal_move() const {
        for (int a = 0; a < 4; ++a) {
            if (is_move_legal(a)) return true;
        }
        return false;
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "+----------------------------+\n";
        for (int r = 0; r < 4; ++r) {
            out << "|";
            for (int c = 0; c < 4; ++c) {
                out << std::setw(7) << tile_value(r * 4 + c);
            }
            out << "|\n";
        }
        out << "+----------------------------+";
        return out.str();
    }

private:
    struct Lookup {
        uint32_t left = 0;
        uint32_t right = 0;
        int left_score = 0;
        int right_score = 0;

        static bool can_merge(int a, int b) {
            if (a == 0 || b == 0) return false;
            if (a == 1 && b == 1) return true;
            return std::abs(a - b) == 1;
        }

        static int merged_code(int a, int b) {
            if (a == 1 && b == 1) return 2;
            return std::max(a, b) + 1;
        }

        static int move_left_codes(int row[4]) {
            int out[4] = {0, 0, 0, 0};
            int top = 0;
            int hold = 0;
            int score = 0;

            for (int i = 0; i < 4; ++i) {
                const int tile = row[i];
                if (tile == 0) continue;

                if (hold == 0) {
                    hold = tile;
                    continue;
                }

                if (can_merge(hold, tile)) {
                    const int merged = merged_code(hold, tile);
                    out[top++] = merged;
                    score += int(Fib2584Board::tile_value_from_code(merged));
                    hold = 0;
                } else {
                    out[top++] = hold;
                    hold = tile;
                }
            }

            if (hold != 0) out[top++] = hold;
            for (int i = 0; i < 4; ++i) row[i] = out[i];
            return score;
        }

        void init(int packed) {
            const int v[4] = {
                (packed >> 0) & Fib2584Board::kCellMask,
                (packed >> 5) & Fib2584Board::kCellMask,
                (packed >> 10) & Fib2584Board::kCellMask,
                (packed >> 15) & Fib2584Board::kCellMask,
            };

            int l[4] = {v[0], v[1], v[2], v[3]};
            int r[4] = {v[3], v[2], v[1], v[0]};

            left_score = move_left_codes(l);
            left = uint32_t((l[0] << 0) | (l[1] << 5) | (l[2] << 10) | (l[3] << 15));

            right_score = move_left_codes(r);
            std::reverse(r, r + 4);
            right = uint32_t((r[0] << 0) | (r[1] << 5) | (r[2] << 10) | (r[3] << 15));
        }

        static const Lookup& find(int packed) {
            static const std::vector<Lookup> cache = []() {
                std::vector<Lookup> table(1u << Fib2584Board::kRowBits);
                for (uint32_t i = 0; i < table.size(); ++i) table[i].init(int(i));
                return table;
            }();
            return cache[packed];
        }
    };

    int move_left() {
        raw_t moved = 0;
        const raw_t prev = raw_;
        int score = 0;
        for (int r = 0; r < 4; ++r) {
            const auto& lu = Lookup::find(fetch_row(r));
            moved |= raw_t(lu.left) << (r * kRowBits);
            score += lu.left_score;
        }
        raw_ = moved;
        return (raw_ != prev) ? score : -1;
    }

    int move_right() {
        raw_t moved = 0;
        const raw_t prev = raw_;
        int score = 0;
        for (int r = 0; r < 4; ++r) {
            const auto& lu = Lookup::find(fetch_row(r));
            moved |= raw_t(lu.right) << (r * kRowBits);
            score += lu.right_score;
        }
        raw_ = moved;
        return (raw_ != prev) ? score : -1;
    }

    int move_up() {
        rotate_clockwise();
        const int score = move_right();
        rotate_counterclockwise();
        return score;
    }

    int move_down() {
        rotate_clockwise();
        const int score = move_left();
        rotate_counterclockwise();
        return score;
    }

    void transpose() {
        Fib2584Board tmp;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                tmp.set_code(c * 4 + r, at_code(r * 4 + c));
            }
        }
        raw_ = tmp.raw_;
    }

    void mirror() {
        Fib2584Board tmp;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                tmp.set_code(r * 4 + (3 - c), at_code(r * 4 + c));
            }
        }
        raw_ = tmp.raw_;
    }

    void flip() {
        Fib2584Board tmp;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                tmp.set_code((3 - r) * 4 + c, at_code(r * 4 + c));
            }
        }
        raw_ = tmp.raw_;
    }

    void rotate_clockwise() {
        transpose();
        mirror();
    }

    void rotate_counterclockwise() {
        transpose();
        flip();
    }

    raw_t raw_;
};

class Fib2584Env {
public:
    explicit Fib2584Env(uint32_t seed = 0) : rng_(seed), score_(0), done_(false), cache_valid_(false), cache_done_(false) {
        reset_internal();
    }

    static Fib2584Env from_state(const py::array& state, int score = 0, py::object seed = py::none()) {
        Fib2584Env env(resolve_seed(seed));
        env.set_state(state, score);
        return env;
    }

    static Fib2584Env from_state_codes(const py::array& state_codes, int score = 0, py::object seed = py::none()) {
        Fib2584Env env(resolve_seed(seed));
        env.set_state_codes(state_codes, score);
        return env;
    }

    Fib2584Env clone() const { return *this; }

    py::array_t<int> reset(py::object seed = py::none()) {
        if (!seed.is_none()) set_seed(seed.cast<uint32_t>());
        reset_internal();
        return get_state();
    }

    py::array_t<uint8_t> reset_codes(py::object seed = py::none()) {
        if (!seed.is_none()) set_seed(seed.cast<uint32_t>());
        reset_internal();
        return get_state_codes();
    }

    void set_state(const py::array& state, int score = 0) {
        const auto values = parse_values_state(state);
        board_.clear();
        for (int i = 0; i < 16; ++i) {
            board_.set_code(i, Fib2584Board::code_from_tile_value(static_cast<uint64_t>(values[i])));
        }
        score_ = score;
        done_ = !board_.has_legal_move();
        invalidate_cache();
    }

    void set_state_codes(const py::array& state_codes, int score = 0) {
        const auto codes = parse_codes_state(state_codes);
        board_.clear();
        for (int i = 0; i < 16; ++i) {
            board_.set_code(i, int(codes[i]));
        }
        score_ = score;
        done_ = !board_.has_legal_move();
        invalidate_cache();
    }

    py::tuple step(int action) {
        validate_action(action);
        int reward = 0;
        bool moved = false;
        if (!done_) {
            reward = board_.move(action);
            moved = reward >= 0;
            if (moved) {
                score_ += reward;
                board_.popup(rng_);
                invalidate_cache();
            } else {
                reward = 0;
            }
            done_ = !board_.has_legal_move();
        }

        py::dict info;
        info["reward"] = reward;
        info["moved"] = moved;
        info["legal_actions"] = py::cast(board_.legal_actions());
        return py::make_tuple(get_state(), score_, done_, info);
    }

    py::tuple step_reward(int action) {
        validate_action(action);
        int reward = 0;
        bool moved = false;
        if (!done_) {
            reward = board_.move(action);
            moved = reward >= 0;
            if (moved) {
                score_ += reward;
                board_.popup(rng_);
                invalidate_cache();
            } else {
                reward = 0;
            }
            done_ = !board_.has_legal_move();
        }

        py::dict info;
        info["moved"] = moved;
        info["score"] = score_;
        info["legal_actions"] = py::cast(board_.legal_actions());
        return py::make_tuple(get_state(), reward, done_, info);
    }

    py::tuple step_reward_codes(int action) {
        validate_action(action);
        int reward = 0;
        bool moved = false;
        if (!done_) {
            const auto raw = board_.raw();
            if (cache_valid_ && cache_raw_ == raw) {
                reward = cache_rewards_[action];
                moved = cache_moved_[action];
                if (moved) {
                    board_ = cache_afterstates_[action];
                    score_ += reward;
                    board_.popup(rng_);
                    done_ = !board_.has_legal_move();
                    invalidate_cache();
                }
            } else {
                reward = board_.move(action);
                moved = reward >= 0;
                if (moved) {
                    score_ += reward;
                    board_.popup(rng_);
                    invalidate_cache();
                    done_ = !board_.has_legal_move();
                } else {
                    reward = 0;
                }
            }
        }
        return py::make_tuple(get_state_codes(), reward, done_, moved);
    }

    py::tuple simulate_step(int action) const {
        validate_action(action);
        Fib2584Board copy = board_;
        int reward = copy.move(action);
        bool moved = reward >= 0;
        if (!moved) reward = 0;
        return py::make_tuple(board_to_array(copy), reward, moved);
    }

    py::tuple simulate_step_codes(int action) const {
        validate_action(action);
        Fib2584Board copy = board_;
        int reward = copy.move(action);
        bool moved = reward >= 0;
        if (!moved) reward = 0;
        return py::make_tuple(board_to_codes_array(copy), reward, moved);
    }

    py::tuple simulate_all_steps_codes() {
        fill_cache();

        py::array_t<uint8_t> states({4, 16});
        auto s = states.mutable_unchecked<2>();
        for (int a = 0; a < 4; ++a) {
            auto codes = cache_afterstates_[a].to_codes_flat();
            for (int i = 0; i < 16; ++i) s(a, i) = codes[i];
        }

        py::array_t<int> rewards({4});
        auto r = rewards.mutable_unchecked<1>();
        py::array_t<uint8_t> moved({4});
        auto m = moved.mutable_unchecked<1>();
        for (int a = 0; a < 4; ++a) {
            r(a) = cache_rewards_[a];
            m(a) = static_cast<uint8_t>(cache_moved_[a] ? 1 : 0);
        }

        return py::make_tuple(states, rewards, moved);
    }

    py::tuple sample_chance() {
        bool spawned = false;
        if (!done_) {
            spawned = board_.popup(rng_);
            done_ = !board_.has_legal_move();
            invalidate_cache();
        }
        return py::make_tuple(get_state(), score_, done_, spawned);
    }

    py::tuple sample_chance_codes() {
        bool spawned = false;
        if (!done_) {
            spawned = board_.popup(rng_);
            done_ = !board_.has_legal_move();
            invalidate_cache();
        }
        return py::make_tuple(get_state_codes(), score_, done_, spawned);
    }

    py::list chance_outcomes() const {
        py::list out;
        const auto empty = empty_positions();
        if (empty.empty()) return out;

        const double inv_n = 1.0 / static_cast<double>(empty.size());
        for (int pos : empty) {
            for (const auto [code, tile_prob] : spawn_distribution()) {
                Fib2584Board next = board_;
                next.set_code(pos, code);
                py::dict item;
                item["state"] = board_to_array(next);
                item["prob"] = tile_prob * inv_n;
                item["position"] = pos;
                item["row"] = pos / 4;
                item["col"] = pos % 4;
                item["tile_code"] = code;
                item["tile_value"] = int(Fib2584Board::tile_value_from_code(code));
                item["score"] = score_;
                item["done"] = !next.has_legal_move();
                out.append(item);
            }
        }
        return out;
    }

    py::list chance_outcomes_codes() const {
        py::list out;
        const auto empty = empty_positions();
        if (empty.empty()) return out;

        const double inv_n = 1.0 / static_cast<double>(empty.size());
        for (int pos : empty) {
            for (const auto [code, tile_prob] : spawn_distribution()) {
                Fib2584Board next = board_;
                next.set_code(pos, code);
                py::dict item;
                item["state_codes"] = board_to_codes_array(next);
                item["prob"] = tile_prob * inv_n;
                item["position"] = pos;
                item["row"] = pos / 4;
                item["col"] = pos % 4;
                item["tile_code"] = code;
                item["tile_value"] = int(Fib2584Board::tile_value_from_code(code));
                item["score"] = score_;
                item["done"] = !next.has_legal_move();
                out.append(item);
            }
        }
        return out;
    }

    bool is_move_legal(int action) const {
        validate_action(action);
        return board_.is_move_legal(action);
    }

    std::vector<int> legal_actions() const {
        return board_.legal_actions();
    }

    py::array_t<uint8_t> legal_action_mask() const {
        py::array_t<uint8_t> arr({4});
        auto m = arr.mutable_unchecked<1>();
        for (int a = 0; a < 4; ++a) m(a) = static_cast<uint8_t>(board_.is_move_legal(a) ? 1 : 0);
        return arr;
    }

    py::array_t<int> get_state() const {
        return board_to_array(board_);
    }

    py::array_t<uint8_t> get_state_codes() const {
        return board_to_codes_array(board_);
    }

    py::array_t<int> board() const { return get_state(); }

    int score() const { return score_; }
    bool done() const { return done_; }
    int max_tile() const { return board_.max_tile_value(); }
    int max_code() const { return board_.max_code(); }
    int empty_count() const { return board_.empty_count(); }
    int board_size() const { return 4; }

    void set_seed(uint32_t seed) { rng_.seed(seed); }
    std::string render() const { return board_.to_string(); }
    std::vector<std::string> action_meanings() const { return {"up", "down", "left", "right"}; }

private:
    static uint32_t resolve_seed(const py::object& seed) {
        if (seed.is_none()) {
            return std::random_device{}();
        }
        return seed.cast<uint32_t>();
    }

    static py::array_t<int> board_to_array(const Fib2584Board& board) {
        auto values = board.to_values_flat();
        py::array_t<int> arr({4, 4});
        auto buf = arr.mutable_unchecked<2>();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                buf(r, c) = values[r * 4 + c];
            }
        }
        return arr;
    }

    static py::array_t<uint8_t> board_to_codes_array(const Fib2584Board& board) {
        auto codes = board.to_codes_flat();
        py::array_t<uint8_t> arr({16});
        auto buf = arr.mutable_unchecked<1>();
        for (int i = 0; i < 16; ++i) buf(i) = codes[i];
        return arr;
    }

    static std::array<int, 16> parse_values_state(const py::array& state) {
        py::array_t<long long, py::array::c_style | py::array::forcecast> arr(state);
        auto info = arr.request();
        std::array<int, 16> values{};
        const auto* ptr = static_cast<const long long*>(info.ptr);

        if (info.ndim == 1) {
            if (info.shape[0] != 16) throw std::invalid_argument("state must have shape (16,) or (4,4)");
            for (int i = 0; i < 16; ++i) values[i] = static_cast<int>(ptr[i]);
            return values;
        }

        if (info.ndim == 2) {
            if (info.shape[0] != 4 || info.shape[1] != 4) throw std::invalid_argument("state must have shape (16,) or (4,4)");
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    values[r * 4 + c] = static_cast<int>(ptr[r * 4 + c]);
                }
            }
            return values;
        }

        throw std::invalid_argument("state must have shape (16,) or (4,4)");
    }

    static std::array<uint8_t, 16> parse_codes_state(const py::array& state) {
        py::array_t<long long, py::array::c_style | py::array::forcecast> arr(state);
        auto info = arr.request();
        std::array<uint8_t, 16> codes{};
        const auto* ptr = static_cast<const long long*>(info.ptr);

        if (info.ndim == 1) {
            if (info.shape[0] != 16) throw std::invalid_argument("state_codes must have shape (16,) or (4,4)");
            for (int i = 0; i < 16; ++i) {
                if (ptr[i] < 0 || ptr[i] > Fib2584Board::kCellMask) throw std::invalid_argument("invalid tile code in state_codes");
                codes[i] = static_cast<uint8_t>(ptr[i]);
            }
            return codes;
        }

        if (info.ndim == 2) {
            if (info.shape[0] != 4 || info.shape[1] != 4) throw std::invalid_argument("state_codes must have shape (16,) or (4,4)");
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    const auto v = ptr[r * 4 + c];
                    if (v < 0 || v > Fib2584Board::kCellMask) throw std::invalid_argument("invalid tile code in state_codes");
                    codes[r * 4 + c] = static_cast<uint8_t>(v);
                }
            }
            return codes;
        }

        throw std::invalid_argument("state_codes must have shape (16,) or (4,4)");
    }

    static void validate_action(int action) {
        if (action < 0 || action > 3) {
            throw std::invalid_argument("action must be one of {0:up, 1:down, 2:left, 3:right}");
        }
    }

    std::vector<int> empty_positions() const {
        std::vector<int> empty;
        empty.reserve(16);
        for (int i = 0; i < 16; ++i) {
            if (board_.at_code(i) == 0) empty.push_back(i);
        }
        return empty;
    }

    static constexpr std::array<std::pair<int, double>, 2> spawn_distribution() {
        return {{{1, 0.9}, {2, 0.1}}};
    }

    void fill_cache() {
        if (done_) return;
        const auto raw = board_.raw();
        if (cache_valid_ && cache_raw_ == raw && cache_done_ == done_) return;
        cache_raw_ = raw;
        cache_done_ = done_;
        for (int a = 0; a < 4; ++a) {
            cache_afterstates_[a] = board_;
            int reward = cache_afterstates_[a].move(a);
            bool moved = reward >= 0;
            if (!moved) reward = 0;
            cache_rewards_[a] = reward;
            cache_moved_[a] = moved;
            if (!moved) cache_afterstates_[a] = board_;
        }
        cache_valid_ = true;
    }

    void invalidate_cache() {
        cache_valid_ = false;
    }

    void reset_internal() {
        score_ = 0;
        done_ = false;
        board_.init(rng_);
        done_ = !board_.has_legal_move();
        invalidate_cache();
    }

    Fib2584Board board_;
    std::mt19937 rng_;
    int score_;
    bool done_;

    bool cache_valid_;
    bool cache_done_;
    Fib2584Board::raw_t cache_raw_{};
    std::array<Fib2584Board, 4> cache_afterstates_{};
    std::array<int, 4> cache_rewards_{};
    std::array<bool, 4> cache_moved_{};
};

PYBIND11_MODULE(fib2584_env, m) {
    m.doc() = "Fast C++ 2584 environment exposed to Python via pybind11";

    py::class_<Fib2584Env>(m, "Fib2584Env")
        .def(py::init<uint32_t>(), py::arg("seed") = 0)
        .def_static("from_state", &Fib2584Env::from_state,
                    py::arg("state"), py::arg("score") = 0, py::arg("seed") = py::none())
        .def_static("from_state_codes", &Fib2584Env::from_state_codes,
                    py::arg("state_codes"), py::arg("score") = 0, py::arg("seed") = py::none())
        .def("clone", &Fib2584Env::clone)
        .def("reset", &Fib2584Env::reset, py::arg("seed") = py::none())
        .def("reset_codes", &Fib2584Env::reset_codes, py::arg("seed") = py::none())
        .def("set_state", &Fib2584Env::set_state, py::arg("state"), py::arg("score") = 0)
        .def("set_state_codes", &Fib2584Env::set_state_codes, py::arg("state_codes"), py::arg("score") = 0)
        .def("step", &Fib2584Env::step, py::arg("action"))
        .def("step_reward", &Fib2584Env::step_reward, py::arg("action"))
        .def("step_reward_codes", &Fib2584Env::step_reward_codes, py::arg("action"))
        .def("simulate_step", &Fib2584Env::simulate_step, py::arg("action"))
        .def("simulate_step_codes", &Fib2584Env::simulate_step_codes, py::arg("action"))
        .def("simulate_all_steps_codes", &Fib2584Env::simulate_all_steps_codes)
        .def("sample_chance", &Fib2584Env::sample_chance)
        .def("sample_chance_codes", &Fib2584Env::sample_chance_codes)
        .def("chance_outcomes", &Fib2584Env::chance_outcomes)
        .def("chance_outcomes_codes", &Fib2584Env::chance_outcomes_codes)
        .def("is_move_legal", &Fib2584Env::is_move_legal, py::arg("action"))
        .def("legal_actions", &Fib2584Env::legal_actions)
        .def("legal_action_mask", &Fib2584Env::legal_action_mask)
        .def("get_state", &Fib2584Env::get_state)
        .def("get_state_codes", &Fib2584Env::get_state_codes)
        .def("board", &Fib2584Env::board)
        .def("set_seed", &Fib2584Env::set_seed, py::arg("seed"))
        .def("render", &Fib2584Env::render)
        .def("action_meanings", &Fib2584Env::action_meanings)
        .def_property_readonly("score", &Fib2584Env::score)
        .def_property_readonly("done", &Fib2584Env::done)
        .def_property_readonly("max_tile", &Fib2584Env::max_tile)
        .def_property_readonly("max_code", &Fib2584Env::max_code)
        .def_property_readonly("empty_count", &Fib2584Env::empty_count)
        .def_property_readonly("size", &Fib2584Env::board_size)
        .def("__copy__", &Fib2584Env::clone)
        .def("__deepcopy__", [](const Fib2584Env& env, py::dict) { return env.clone(); })
        .def("__repr__", [](const Fib2584Env& env) {
            std::ostringstream oss;
            oss << "Fib2584Env(score=" << env.score()
                << ", done=" << (env.done() ? "True" : "False")
                << ", max_tile=" << env.max_tile() << ")\n"
                << env.render();
            return oss.str();
        });
}
