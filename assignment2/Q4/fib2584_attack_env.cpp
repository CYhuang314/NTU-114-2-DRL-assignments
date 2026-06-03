#include <pybind11/pybind11.h>
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

    int tile_value(int idx) const {
        return int(tile_value_from_code(at_code(idx)));
    }

    int empty_count() const {
        int count = 0;
        for (int i = 0; i < 16; ++i) count += (at_code(i) == 0);
        return count;
    }

    std::vector<int> empty_positions() const {
        std::vector<int> out;
        out.reserve(16);
        for (int i = 0; i < 16; ++i) {
            if (at_code(i) == 0) out.push_back(i);
        }
        return out;
    }

    std::vector<int> to_values_flat() const {
        std::vector<int> values(16);
        for (int i = 0; i < 16; ++i) values[i] = tile_value(i);
        return values;
    }

    std::vector<int> to_codes_flat() const {
        std::vector<int> codes(16);
        for (int i = 0; i < 16; ++i) codes[i] = at_code(i);
        return codes;
    }

    std::vector<std::vector<int>> to_values_grid() const {
        std::vector<std::vector<int>> grid(4, std::vector<int>(4, 0));
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                grid[r][c] = tile_value(r * 4 + c);
            }
        }
        return grid;
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

    bool place_tile(int pos, int code) {
        if (pos < 0 || pos >= 16) throw std::invalid_argument("position must be in [0, 15]");
        if (code != 1 && code != 2) throw std::invalid_argument("tile code must be 1 or 2");
        if (at_code(pos) != 0) return false;
        set_code(pos, code);
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
        return (raw_ == prev) ? -1 : score;
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
        return (raw_ == prev) ? -1 : score;
    }

    int move_up() {
        const Fib2584Board prev = *this;
        int score = 0;
        for (int c = 0; c < 4; ++c) {
            int row[4] = {prev.at_code(c), prev.at_code(4 + c), prev.at_code(8 + c), prev.at_code(12 + c)};
            score += Lookup::move_left_codes(row);
            set_code(c, row[0]);
            set_code(4 + c, row[1]);
            set_code(8 + c, row[2]);
            set_code(12 + c, row[3]);
        }
        return (raw_ == prev.raw_) ? -1 : score;
    }

    int move_down() {
        const Fib2584Board prev = *this;
        int score = 0;
        for (int c = 0; c < 4; ++c) {
            int row[4] = {prev.at_code(12 + c), prev.at_code(8 + c), prev.at_code(4 + c), prev.at_code(c)};
            score += Lookup::move_left_codes(row);
            set_code(12 + c, row[0]);
            set_code(8 + c, row[1]);
            set_code(4 + c, row[2]);
            set_code(c, row[3]);
        }
        return (raw_ == prev.raw_) ? -1 : score;
    }

    raw_t raw_;
};

class Fib2584AttackEnv {
public:
    explicit Fib2584AttackEnv(uint32_t seed = 0) : rng_(seed) {
        reset_internal();
    }

    Fib2584AttackEnv clone() const { return *this; }

    py::dict reset(py::object seed = py::none()) {
        if (!seed.is_none()) rng_.seed(seed.cast<uint32_t>());
        reset_internal();
        return observation();
    }

    py::dict observation() const {
        py::dict obs;
        obs["board_a"] = board_a_.to_values_grid();
        obs["board_b"] = board_b_.to_values_grid();
        obs["board_a_codes"] = board_a_.to_codes_flat();
        obs["board_b_codes"] = board_b_.to_codes_flat();
        obs["current_player"] = current_player_;
        obs["current_player_name"] = current_player_ == 0 ? "A" : "B";
        obs["phase"] = done_ ? "done" : "slide";
        obs["phase_id"] = done_ ? 1 : 0;
        obs["turn_index"] = turn_index_;
        obs["done"] = done_;
        obs["winner"] = winner_;
        obs["winner_name"] = winner_ == -1 ? py::none() : py::cast(winner_ == 0 ? "A" : "B");
        obs["loser"] = loser_;
        obs["loser_name"] = loser_ == -1 ? py::none() : py::cast(loser_ == 0 ? "A" : "B");
        obs["legal_slide_actions"] = legal_slide_actions();
        obs["legal_tile_codes"] = std::vector<int>{1, 2};
        obs["last_slide_action"] = last_slide_action_;
        obs["last_slide_reward_a"] = last_slide_reward_a_;
        obs["last_slide_reward_b"] = last_slide_reward_b_;
        obs["last_slide_moved_a"] = last_slide_moved_a_;
        obs["last_slide_moved_b"] = last_slide_moved_b_;
        obs["empty_positions_a"] = board_a_.empty_positions();
        obs["empty_positions_b"] = board_b_.empty_positions();
        return obs;
    }

    std::vector<int> legal_slide_actions() const {
        if (done_) return {};
        return current_board().legal_actions();
    }

    py::dict preview_turn(int action) const {
        if (done_) throw std::logic_error("cannot preview after game is done");
        if (!current_board().is_move_legal(action)) {
            throw std::invalid_argument("slide action is illegal on current player's board");
        }

        Fib2584Board ba = board_a_;
        Fib2584Board bb = board_b_;

        int reward_a = ba.move(action);
        bool moved_a = reward_a >= 0;
        if (!moved_a) reward_a = 0;

        int reward_b = bb.move(action);
        bool moved_b = reward_b >= 0;
        if (!moved_b) reward_b = 0;

        py::dict out;
        out["slide_action"] = action;
        out["board_a_after_slide"] = ba.to_values_grid();
        out["board_b_after_slide"] = bb.to_values_grid();
        out["board_a_codes_after_slide"] = ba.to_codes_flat();
        out["board_b_codes_after_slide"] = bb.to_codes_flat();
        out["empty_positions_a"] = ba.empty_positions();
        out["empty_positions_b"] = bb.empty_positions();
        out["must_skip_place_a"] = (ba.empty_count() == 0);
        out["must_skip_place_b"] = (bb.empty_count() == 0);
        out["slide_reward_a"] = reward_a;
        out["slide_reward_b"] = reward_b;
        out["slide_moved_a"] = moved_a;
        out["slide_moved_b"] = moved_b;
        return out;
    }

    py::dict step_turn(int action, py::object place_a = py::none(), py::object place_b = py::none()) {
        if (done_) throw std::logic_error("cannot act after game is done");
        if (!current_board().is_move_legal(action)) {
            throw std::invalid_argument("slide action is illegal on current player's board");
        }

        last_slide_action_ = action;
        last_slide_reward_a_ = board_a_.move(action);
        last_slide_moved_a_ = last_slide_reward_a_ >= 0;
        if (!last_slide_moved_a_) last_slide_reward_a_ = 0;

        last_slide_reward_b_ = board_b_.move(action);
        last_slide_moved_b_ = last_slide_reward_b_ >= 0;
        if (!last_slide_moved_b_) last_slide_reward_b_ = 0;

        apply_placement(0, place_a);
        apply_placement(1, place_b);
        finalize_turn();
        return observation();
    }

    std::string render() const {
        std::ostringstream oss;
        oss << "Turn: " << turn_index_ << " | Current player: " << (current_player_ == 0 ? "A" : "B");
        if (done_) oss << " | Winner: " << (winner_ == 0 ? "A" : "B");
        oss << "\n[Board A]\n" << board_a_.to_string() << "\n\n[Board B]\n" << board_b_.to_string();
        return oss.str();
    }

    bool done() const { return done_; }
    int winner() const { return winner_; }
    int loser() const { return loser_; }
    int current_player() const { return current_player_; }
    int turn_index() const { return turn_index_; }
    std::string phase() const { return done_ ? "done" : "slide"; }

private:
    void reset_internal() {
        board_a_.init(rng_);
        board_b_.init(rng_);
        current_player_ = 0;
        turn_index_ = 0;
        done_ = false;
        winner_ = -1;
        loser_ = -1;
        last_slide_action_ = -1;
        last_slide_reward_a_ = 0;
        last_slide_reward_b_ = 0;
        last_slide_moved_a_ = false;
        last_slide_moved_b_ = false;
        resolve_terminal_before_turn();
    }

    const Fib2584Board& current_board() const {
        return current_player_ == 0 ? board_a_ : board_b_;
    }

    const Fib2584Board& other_board() const {
        return current_player_ == 0 ? board_b_ : board_a_;
    }

    Fib2584Board& board_by_index(int board_index) {
        if (board_index == 0) return board_a_;
        if (board_index == 1) return board_b_;
        throw std::invalid_argument("board_index must be 0 or 1");
    }

    const Fib2584Board& board_by_index(int board_index) const {
        if (board_index == 0) return board_a_;
        if (board_index == 1) return board_b_;
        throw std::invalid_argument("board_index must be 0 or 1");
    }

    void apply_placement(int board_index, const py::object& obj) {
        Fib2584Board& board = board_by_index(board_index);
        if (board.empty_count() == 0) {
            if (!obj.is_none()) {
                throw std::invalid_argument("placement must be None when target board has no empty cells");
            }
            return;
        }
        if (obj.is_none()) {
            throw std::invalid_argument("placement cannot be None when target board still has empty cells");
        }
        py::dict d = obj.cast<py::dict>();
        if (!d.contains("position") || !d.contains("tile_code")) {
            throw std::invalid_argument("placement dict must contain 'position' and 'tile_code'");
        }
        const int position = d["position"].cast<int>();
        const int tile_code = d["tile_code"].cast<int>();
        if (!board.place_tile(position, tile_code)) {
            throw std::invalid_argument("invalid placement: non-empty position or invalid tile code");
        }
    }

    void resolve_terminal_before_turn() {
        const bool cur_can = current_board().has_legal_move();
        const bool other_can = other_board().has_legal_move();
        if (cur_can || other_can) return;
        done_ = true;
        winner_ = 1 - current_player_;
        loser_ = current_player_;
    }

    void finalize_turn() {
        const bool a_can = board_a_.has_legal_move();
        const bool b_can = board_b_.has_legal_move();

        if (!a_can && !b_can) {
            done_ = true;
            winner_ = 1 - current_player_;
            loser_ = current_player_;
            return;
        }
        if (a_can && !b_can) {
            done_ = true;
            winner_ = 0;
            loser_ = 1;
            return;
        }
        if (!a_can && b_can) {
            done_ = true;
            winner_ = 1;
            loser_ = 0;
            return;
        }

        current_player_ = 1 - current_player_;
        ++turn_index_;
        resolve_terminal_before_turn();
    }

    Fib2584Board board_a_;
    Fib2584Board board_b_;
    std::mt19937 rng_;

    int current_player_ = 0;
    int turn_index_ = 0;
    bool done_ = false;
    int winner_ = -1;
    int loser_ = -1;

    int last_slide_action_ = -1;
    int last_slide_reward_a_ = 0;
    int last_slide_reward_b_ = 0;
    bool last_slide_moved_a_ = false;
    bool last_slide_moved_b_ = false;
};

PYBIND11_MODULE(fib2584_attack_env, m) {
    m.doc() = "Fib-2584 attack environment with single-turn API";

    py::class_<Fib2584AttackEnv>(m, "Fib2584AttackEnv")
        .def(py::init<uint32_t>(), py::arg("seed") = 0)
        .def("clone", &Fib2584AttackEnv::clone)
        .def("reset", &Fib2584AttackEnv::reset, py::arg("seed") = py::none())
        .def("observation", &Fib2584AttackEnv::observation)
        .def("legal_slide_actions", &Fib2584AttackEnv::legal_slide_actions)
        .def("preview_turn", &Fib2584AttackEnv::preview_turn, py::arg("action"))
        .def("step_turn", &Fib2584AttackEnv::step_turn,
             py::arg("action"), py::arg("place_a") = py::none(), py::arg("place_b") = py::none())
        .def("render", &Fib2584AttackEnv::render)
        .def_property_readonly("done", &Fib2584AttackEnv::done)
        .def_property_readonly("winner", &Fib2584AttackEnv::winner)
        .def_property_readonly("loser", &Fib2584AttackEnv::loser)
        .def_property_readonly("current_player", &Fib2584AttackEnv::current_player)
        .def_property_readonly("turn_index", &Fib2584AttackEnv::turn_index)
        .def_property_readonly("phase", &Fib2584AttackEnv::phase)
        .def("__copy__", &Fib2584AttackEnv::clone)
        .def("__deepcopy__", [](const Fib2584AttackEnv& env, py::dict) { return env.clone(); })
        .def("__repr__", [](const Fib2584AttackEnv& env) {
            std::ostringstream oss;
            oss << "Fib2584AttackEnv(done=" << (env.done() ? "True" : "False")
                << ", winner=" << env.winner() << ", current_player=" << env.current_player()
                << ", turn_index=" << env.turn_index() << ")\n"
                << env.render();
            return oss.str();
        });
}
