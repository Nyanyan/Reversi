#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

// Reversi AI C++ version

#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <chrono>
#include <utility>
#include <string>
#include <cmath>
#include <map>
#include <unordered_map>
#include <random>
#include <time.h>

using namespace std;

#define hw 8
#define hw2 64
#define window 0.00001

int dy[8] = {0, 1, 0, -1, 1, 1, -1, -1};
int dx[8] = {1, 0, -1, 0, 1, -1, 1, -1};

int tim(){
    return static_cast<int>(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count());
}

struct HashPair {
    static size_t m_hash_pair_random;
    template<class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);
        size_t seed = 0;
        seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= m_hash_pair_random + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

size_t HashPair::m_hash_pair_random = (size_t) random_device()();

double weight[hw2] = {
    3.2323232323232323, 0.23088023088023088, 1.3852813852813852, 1.0389610389610389, 1.0389610389610389, 1.3852813852813852, 0.23088023088023088, 3.2323232323232323,
    0.23088023088023088, 0.0, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.0, 0.23088023088023088,
    1.3852813852813852, 0.9004329004329005, 1.0389610389610389, 0.9466089466089466, 0.9466089466089466, 1.0389610389610389, 0.9004329004329005, 1.3852813852813852,
    1.0389610389610389, 0.9004329004329005, 0.9466089466089466, 0.9235209235209235, 0.9235209235209235, 0.9466089466089466, 0.9004329004329005, 1.0389610389610389,
    1.0389610389610389, 0.9004329004329005, 0.9466089466089466, 0.9235209235209235, 0.9235209235209235, 0.9466089466089466, 0.9004329004329005, 1.0389610389610389,
    1.3852813852813852, 0.9004329004329005, 1.0389610389610389, 0.9466089466089466, 0.9466089466089466, 1.0389610389610389, 0.9004329004329005, 1.3852813852813852,
    0.23088023088023088, 0.0, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.0, 0.23088023088023088,
    3.2323232323232323, 0.23088023088023088, 1.3852813852813852, 1.0389610389610389, 1.0389610389610389, 1.3852813852813852, 0.23088023088023088, 3.2323232323232323
};

int confirm_lst[hw][hw] = {
    {63, 62, 61, 60, 59, 58, 57, 56},
    {56, 57, 58, 59, 60, 61, 62, 63},
    {63, 55, 47, 39, 31, 23, 15,  7},
    { 7, 15, 23, 31, 39, 47, 55, 63},
    { 7,  6,  5,  4,  3,  2,  1,  0},
    { 0,  1,  2,  3,  4,  5,  6,  7},
    {56, 48, 40 , 32, 24, 16, 8,  0},
    { 0,  8, 16, 24, 32, 40, 48, 56}
};

unsigned long long confirm_num[4] = {
    0b0000000000000000000000000000000000000000000000000000000011111111,
    0b0000000100000001000000010000000100000001000000010000000100000001,
    0b1111111100000000000000000000000000000000000000000000000000000000,
    0b1000000010000000100000001000000010000000100000001000000010000000
};

struct grid_priority{
    double priority;
    unsigned long long me;
    unsigned long long op;
    int open_val;
};

int ai_player;
double weight_weight, canput_weight, confirm_weight, stone_weight, open_weight;
int max_depth, vacant_cnt;
double game_ratio;
unordered_map<pair<unsigned long long, unsigned long long>, double, HashPair> memo;
unsigned long long marked;
int min_max_depth;
int tl, strt;

unsigned long long check_mobility(unsigned long long grid_me, unsigned long long grid_op){
    unsigned long long res, t, u, v, w;
    res = 0;
    // 左だけ高速に演算できる
    t = 0b1111111100000000111111110000000011111111000000001111111100000000;
    u = grid_op & t;
    v = (grid_me & t) << 1;
    w = ~(v | grid_op);
    res |= w & (u + v) & t;
    t = 0b0000000011111111000000001111111100000000111111110000000011111111;
    u = grid_op & t;
    v = (grid_me & t) << 1;
    w = ~(v | grid_op);
    res |= w & (u + v) & t;

    w = grid_op & 0x7e7e7e7e7e7e7e7e;
    t = w & (grid_me >> 1);
    t |= w & (t >> 1);
    t |= w & (t >> 1);
    t |= w & (t >> 1);
    t |= w & (t >> 1);
    t |= w & (t >> 1);
    res |= (t >> 1);

    w = grid_op & 0x00FFFFFFFFFFFF00;
    t = w & (grid_me << hw);
    t |= w & (t << hw);
    t |= w & (t << hw);
    t |= w & (t << hw);
    t |= w & (t << hw);
    t |= w & (t << hw);
    res |= (t << hw);
    t = w & (grid_me >> hw);
    t |= w & (t >> hw);
    t |= w & (t >> hw);
    t |= w & (t >> hw);
    t |= w & (t >> hw);
    t |= w & (t >> hw);
    res |= (t >> hw);

    w = grid_op & 0x007e7e7e7e7e7e00;
    t = w & (grid_me << (hw - 1));
    t |= w & (t << (hw - 1));
    t |= w & (t << (hw - 1));
    t |= w & (t << (hw - 1));
    t |= w & (t << (hw - 1));
    t |= w & (t << (hw - 1));
    res |= (t << (hw - 1));
    t = w & (grid_me >> (hw - 1));
    t |= w & (t >> (hw - 1));
    t |= w & (t >> (hw - 1));
    t |= w & (t >> (hw - 1));
    t |= w & (t >> (hw - 1));
    t |= w & (t >> (hw - 1));
    res |= (t >> (hw - 1));

    t = w & (grid_me << (hw + 1));
    t |= w & (t << (hw + 1));
    t |= w & (t << (hw + 1));
    t |= w & (t << (hw + 1));
    t |= w & (t << (hw + 1));
    t |= w & (t << (hw + 1));
    res |= (t << (hw + 1));
    t = w & (grid_me >> (hw + 1));
    t |= w & (t >> (hw + 1));
    t |= w & (t >> (hw + 1));
    t |= w & (t >> (hw + 1));
    t |= w & (t >> (hw + 1));
    t |= w & (t >> (hw + 1));
    res |= (t >> (hw + 1));
    return ~(grid_me | grid_op) & res;
}

int check_confirm(unsigned long long grid, int idx){
    int i, res = 0;
    for (i = 0; i < hw; i++){
        if (1 & (grid >> confirm_lst[idx][i]))
            res++;
        else
            break;
    }
    return res;
}

double evaluate(unsigned long long grid_me, unsigned long long grid_op, int canput, int open_val){
    int canput_all = canput;
    double weight_me = 0.0, weight_op = 0.0;
    int me_cnt = 0, op_cnt = 0;
    int confirm_me = 0, confirm_op = 0;
    int stone_me = 0, stone_op = 0;
    unsigned long long mobility, stones;
    int i, j;
    for (i = 0; i < hw2; i++){
        if (1 & (grid_me >> (hw2 - i - 1))){
            weight_me += weight[i];
            me_cnt++;
        } else if (1 & (grid_op >> (hw2 - i - 1))){
            weight_op += weight[i];
            op_cnt++;
        }
    }
    mobility = check_mobility(grid_me, grid_op);
    for (i = 0; i < hw2; i++)
        canput_all += 1 & (mobility >> i);
    stones = grid_me | grid_op;
    for (i = 0; i < hw; i += 2){
        if (stones ^ confirm_num[i / 2]){
            for (j = 0; j < 2; j++){
                confirm_me += max(0, check_confirm(grid_me, i + j) - 1);
                confirm_op += max(0, check_confirm(grid_op, i + j) - 1);
            }
        } else {
            for (j = 1; j < hw - 1; j++){
                if (1 & (grid_me >> confirm_lst[i][j]))
                    confirm_me++;
                else if (1 & (grid_op >> confirm_lst[i][j]))
                    confirm_op++;
            }
        }
    }
    confirm_me += 1 & grid_me;
    confirm_me += 1 & (grid_me >> (hw - 1));
    confirm_me += 1 & (grid_me >> (hw2 - hw));
    confirm_me += 1 & (grid_me >> (hw2 - 1));
    confirm_op += 1 & grid_op;
    confirm_op += 1 & (grid_op >> (hw - 1));
    confirm_op += 1 & (grid_op >> (hw2 - hw));
    confirm_op += 1 & (grid_op >> (hw2 - 1));
    for (i = 0; i < hw2; i++){
        stone_me += 1 & (grid_me >> i);
        stone_op += 1 & (grid_op >> i);
    }
    double weight_proc, canput_proc, confirm_proc, stone_proc, open_proc;
    weight_proc = weight_me / me_cnt - weight_op / op_cnt;
    canput_proc = (double)(canput_all - canput) / max(1, canput_all) - (double)canput / max(1, canput_all);
    confirm_proc = (double)confirm_me / max(1, confirm_me + confirm_op) - (double)confirm_op / max(1, confirm_me + confirm_op);
    stone_proc = -(double)stone_me / (stone_me + stone_op) + (double)stone_op / (stone_me + stone_op);
    open_proc = max(-1.0, (double)(5 - open_val) / 5);
    return weight_proc * weight_weight + canput_proc * canput_weight + confirm_proc * confirm_weight + stone_proc * stone_weight + open_proc * open_weight;
}

double end_game(unsigned long long grid_me, unsigned long long grid_op){
    int res = 0, i;
    for (i = 0; i < hw2; i++){
        res += 1 & (grid_me >> i);
        res -= 1 & (grid_op >> i);
    }
    return (double)res;
}

unsigned long long transfer(unsigned long long put, int k){
    switch(k){
        case 0:
            return (put << 8) & 0xffffffffffffff00;
        case 1:
            return (put << 7) & 0x7f7f7f7f7f7f7f00;
        case 2:
            return (put >> 1) & 0x7f7f7f7f7f7f7f7f;
        case 3:
            return (put >> 9) & 0x007f7f7f7f7f7f7f;
        case 4:
            return (put >> 8) & 0x00ffffffffffffff;
        case 5:
            return (put >> 7) & 0x00fefefefefefefe;
        case 6:
            return (put << 1) & 0xfefefefefefefefe;
        case 7:
            return (put << 9) & 0xfefefefefefefe00;
    }
    return 0;
}

unsigned long long move(unsigned long long grid_me, unsigned long long grid_op, int place){
    unsigned long long put, rev1, rev2, mask;
    int i;
    put = (unsigned long long)1 << place;
    rev1 = 0;
    for (i = 0; i < hw; i++){
        rev2 = 0;
        mask = transfer(put, i);
        while (mask != 0 && (mask & grid_op) != 0){
            rev2 |= mask;
            mask = transfer(mask, i);
        }
        if ((mask & grid_me) != 0)
            rev1 |= rev2;
    }
    return grid_me ^ (put | rev1);
}

int calc_open(unsigned long long stones, unsigned long long rev){
    int i, res = 0;
    for (i = 0; i < hw2; i++){
        if (1 & (rev >> i)){
            res += 1 - (1 & (stones >> (i + 1)));
            res += 1 - (1 & (stones >> (i - 1)));
            res += 1 - (1 & (stones >> (i + hw)));
            res += 1 - (1 & (stones >> (i - hw)));
            res += 1 - (1 & (stones >> (i + hw + 1)));
            res += 1 - (1 & (stones >> (i - hw + 1)));
            res += 1 - (1 & (stones >> (i + hw - 1)));
            res += 1 - (1 & (stones >> (i - hw - 1)));
        }
    }
    return res;
}

int cmp(grid_priority p, grid_priority q){
    return p.priority > q.priority;
}

double nega_scout(unsigned long long grid_me, unsigned long long grid_op, int depth, double alpha, double beta, int skip_cnt, int canput, int open_val){
    if (max_depth > min_max_depth && tim() - strt > tl)
        return -100000000.0;
    if (skip_cnt == 2)
        return end_game(grid_me, grid_op);
    else if (depth == 0)
        return evaluate(grid_me, grid_op, canput, open_val);
    int y, x, i;
    double val;
    int n_canput = 0;
    unsigned long long mobility = check_mobility(grid_me, grid_op);
    unsigned long long n_grid_me, n_grid_op;
    double priority;
    vector<grid_priority> lst;
    pair<unsigned long long, unsigned long long> grid_all, n_grid_all;
    grid_all.first = grid_me;
    grid_all.second = grid_op;
    for (i = 0; i < hw2; i++){
        if (1 & (mobility >> i)){
            n_canput++;
            n_grid_me = move(grid_me, grid_op, i);
            n_grid_op = (n_grid_me ^ grid_op) & grid_op;
            n_grid_all.first = n_grid_op;
            n_grid_all.second = n_grid_me;
            priority = memo[n_grid_all];
            open_val = calc_open(n_grid_me | n_grid_op, n_grid_me ^ grid_me);
            priority -= 0.1 * open_val;
            grid_priority tmp;
            tmp.priority = priority;
            tmp.me = n_grid_me;
            tmp.op = n_grid_op;
            tmp.open_val = open_val;
            lst.push_back(tmp);
        }
    }
    if (n_canput == 0){
        val = -nega_scout(grid_op, grid_me, depth, -beta, -alpha, skip_cnt + 1, 0, 0);
        if (fabs(val) == 100000000.0)
            return -100000000.0;
        return max(alpha, val);
    }
    if (n_canput > 1)
        sort(lst.begin(), lst.end(), cmp);
    open_val = 0;
    if (depth == 1)
        open_val = lst[0].open_val;
    val = -nega_scout(lst[0].op, lst[0].me, depth - 1, -beta, -alpha, 0, n_canput, open_val);
    if (fabs(val) == 100000000.0)
        return -100000000.0;
    memo[grid_all] = val;
    alpha = max(alpha, val);
    if (alpha >= beta)
        return alpha;
    for (i = 1; i < n_canput; i++){
        if (depth == 1)
            open_val = lst[i].open_val;
        val = -nega_scout(lst[i].op, lst[i].me, depth - 1, -alpha - window, -alpha, 0, n_canput, open_val);
        if (fabs(val) == 100000000.0)
            return -100000000.0;
        if (beta <= val)
            return val;
        if (alpha < val){
            alpha = val;
            val = -nega_scout(lst[i].op, lst[i].me, depth - 1, -beta, -alpha, 0, n_canput, open_val);
            if (fabs(val) == 100000000.0)
                return -100000000.0;
            memo[grid_all] = val;
            alpha = max(alpha, val);
            if (alpha >= beta)
                return alpha;
        }
    }
    return alpha;
}

double map_double(double s, double e, double x){
    return s + (e - s) * x;
}

int main(){
    int ansy, ansx, outy, outx, i, canput, former_depth = 9, former_vacant = hw2 - 4;
    double score, max_score;
    double weight_weight_s, canput_weight_s, confirm_weight_s, stone_weight_s, open_weight_s, weight_weight_e, canput_weight_e, confirm_weight_e, stone_weight_e, open_weight_e;
    unsigned long long in_grid_me, in_grid_op, in_mobility, grid_me, grid_op;
    vector<grid_priority> lst;
    pair<unsigned long long, unsigned long long> grid_all;
    int elem;
    cin >> ai_player;
    cin >> tl;
    cin >> weight_weight_s;
    cin >> canput_weight_s;
    cin >> confirm_weight_s;
    cin >> stone_weight_s;
    cin >> open_weight_s;
    cin >> weight_weight_e;
    cin >> canput_weight_e;
    cin >> confirm_weight_e;
    cin >> stone_weight_e;
    cin >> open_weight_e;
    if (ai_player == 0)
        cerr << "AI initialized AI is Black" << endl;
    else
        cerr << "AI initialized AI is White" << endl;
    while (true){
        outy = -1;
        outx = -1;
        vacant_cnt = 0;
        in_grid_me = 0;
        in_grid_op = 0;
        in_mobility = 0;
        canput = 0;
        for (i = 0; i < hw2; i++){
            cin >> elem;
            vacant_cnt += (int)(elem == -1 || elem == 2);
            in_mobility <<= 1;
            in_mobility += (int)(elem == 2);
            canput += (int)(elem == 2);
            in_grid_me <<= 1;
            in_grid_op <<= 1;
            in_grid_me += (int)(elem == ai_player);
            in_grid_op += (int)(elem == 1 - ai_player);
        }
        if (vacant_cnt > 15)
            min_max_depth = max(5, former_depth + vacant_cnt - former_vacant);
        else
            min_max_depth = 15;
        cerr << "start depth " << min_max_depth << endl;
        max_depth = min_max_depth;
        former_vacant = vacant_cnt;
        lst.clear();
        for (i = 0; i < hw2; i++){
            if (1 & (in_mobility >> i)){
                grid_me = move(in_grid_me, in_grid_op, i);
                grid_op = (grid_me ^ in_grid_op) & in_grid_op;
                grid_all.first = grid_me;
                grid_all.second = grid_op;
                grid_priority tmp;
                tmp.priority = memo[grid_all];
                tmp.me = grid_me;
                tmp.op = grid_op;
                tmp.open_val = i;
                lst.push_back(tmp);
            }
        }
        memo.clear();
        strt = tim();
        while (tim() - strt < tl / 2){
            if (canput > 1)
                sort(lst.begin(), lst.end(), cmp);
            game_ratio = (double)(hw2 - vacant_cnt + max_depth) / hw2;
            weight_weight = map_double(weight_weight_s, weight_weight_e, game_ratio);
            canput_weight = map_double(canput_weight_s, canput_weight_e, game_ratio);
            confirm_weight = map_double(confirm_weight_s, confirm_weight_e, game_ratio);
            stone_weight = map_double(stone_weight_s, stone_weight_e, game_ratio);
            open_weight = map_double(open_weight_s, open_weight_e, game_ratio);
            max_score = -65.0;
            for (i = 0; i < canput; i++){
                grid_me = lst[i].me;
                grid_op = lst[i].op;
                score = -nega_scout(grid_op, grid_me, max_depth - 1, -65.0, -max_score, 0, canput, 0);
                if (fabs(score) == 100000000.0){
                    max_score = -100000000.0;
                    break;
                }
                lst[i].priority = score;
                if (score > max_score){
                    max_score = score;
                    ansy = (hw2 - lst[i].open_val - 1) / hw;
                    ansx = (hw2 - lst[i].open_val - 1) % hw;
                }
            }
            if (max_score == -100000000.0){
                cerr << "depth " << max_depth << " timeoout" << endl;
                break;
            }
            former_depth = max_depth;
            outy = ansy;
            outx = ansx;
            cerr << "depth " << max_depth << " next " << outy << outx << " score " << max_score << " time " << tim() - strt << endl;
            if (vacant_cnt < max_depth || fabs(max_score) >= 1.0){
                cerr << "game end" << endl;
                break;
            }
            max_depth++;
        }
        cout << outy << " " << outx << endl;
    }
    return 0;
}