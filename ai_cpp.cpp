#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx")

// Reversi AI C++ version 2

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
#include <immintrin.h>

using namespace std;

#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw2_m1 63
#define hw2_mhw 56
#define window 0.00001
#define simple_threshold 2
#define inf 1000000000000.0

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

struct eval_param{
    double weight_s[hw2];
    double weight_e[hw2];
    double weight[hw2];
    double weight_weight, canput_weight, confirm_weight, stone_weight, open_weight, out_weight;
    double weight_se[12];
    double open_val_threshold;
    //double weight_weight_s, canput_weight_s, confirm_weight_s, stone_weight_s, open_weight_s, out_weight_s, weight_weight_e, canput_weight_e, confirm_weight_e, stone_weight_e, open_weight_e, out_weight_e;
};

struct confirm_param{
    int lst[hw][hw];
    unsigned long long num[4];
};

typedef union {
	unsigned long long ull[4];
	#ifdef __AVX2__
		__m256i	v4;
	#endif
	__m128i	v2[2];
} V4DI;

struct move_param{
    V4DI lmask_v4[hw2];
    V4DI rmask_v4[hw2];
};

struct search_param{
    unordered_map<pair<unsigned long long, unsigned long long>, double, HashPair> memo1, memo2; 
    unordered_map<pair<unsigned long long, unsigned long long>, double, HashPair> memo_lb, memo_ub;
    int max_depth;
    int min_max_depth;
    int strt, tl;
};

struct grid_priority{
    unsigned long long p;
    unsigned long long o;
    int open_val;
};

struct grid_priority_main{
    double priority;
    unsigned long long p;
    unsigned long long o;
    int move;
    int open_val;
};

eval_param eval_param;
confirm_param confirm_param;
move_param move_param;
search_param search_param;

#ifdef _MSC_VER
	#define	mirror_v(x)	_byteswap_uint64(x)
#else
	#define	mirror_v(x)	__builtin_bswap64(x)
#endif

int tim(){
    return static_cast<int>(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count());
}

void print_board(unsigned long long p, unsigned long long o){
    int i, j, idx;
    for (i = 0; i < hw; i++){
        for (j = 0; j < hw; j++){
            idx = hw2 - i * hw + j;
            if (1 & (p >> idx)){
                cerr << "P ";
            } else if (1 & (o >> idx)){
                cerr << "O ";
            } else {
                cerr << ". ";
            }
        }
        cerr << endl;
    }
    cerr << endl;
}

void init(int argc, char* argv[]){
    FILE *fp;
    char* file;
    if (argc > 1)
        file = argv[1];
    else
        file = "param.txt";
    if ((fp = fopen(file, "r")) == NULL){
        printf("param.txt not exist");
        exit(1);
    }
    char cbuf[1024];
    int translate[hw2] = {
        0, 1, 2, 3, 3, 2, 1, 0,
        1, 4, 5, 6, 6, 5, 4, 1,
        2, 5, 7, 8, 8, 7, 5, 2,
        3, 6, 8, 9, 9, 8, 6, 3,
        3, 6, 8, 9, 9, 8, 6, 3,
        2, 5, 7, 8, 8, 7, 5, 2,
        1, 4, 5, 6, 6, 5, 4, 1,
        0, 1, 2, 3, 3, 2, 1, 0
    };
    double weight_buf[10];
    int i, j;
    for (i = 0; i < 10; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        weight_buf[i] = atof(cbuf);
    }
    for (i = 0; i < hw2; i++)
        eval_param.weight_s[i] = weight_buf[translate[i]];
    for (i = 0; i < 10; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        weight_buf[i] = atof(cbuf);
    }
    for (i = 0; i < hw2; i++)
        eval_param.weight_e[i] = weight_buf[translate[i]];
    for (i = 0; i < 12; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.weight_se[i] = atof(cbuf);
    }
    if ((fp = fopen("const.txt", "r")) == NULL){
        printf("const.txt not exist");
        exit(1);
    }
    for (i = 0; i < hw; i++){
        for (j = 0; j < hw; j++){
            if (!fgets(cbuf, 1024, fp)){
                printf("const.txt broken");
                exit(1);
            }
            confirm_param.lst[i][j] = atoi(cbuf);
        }
    }
    for (i = 0; i < 4; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("const.txt broken");
            exit(1);
        }
        confirm_param.num[i] = stoull(cbuf);
    }
    for (i = 0; i < hw2; i++){
        for (j = 0; j < 4; j++){
            if (!fgets(cbuf, 1024, fp)){
                printf("const.txt broken");
                exit(1);
            }
            move_param.lmask_v4[i].ull[j] = stoull(cbuf);
        }
    }
    for (i = 0; i < hw2; i++){
        for (j = 0; j < 4; j++){
            if (!fgets(cbuf, 1024, fp)){
                printf("const.txt broken");
                exit(1);
            }
            move_param.rmask_v4[i].ull[j] = stoull(cbuf);
        }
    }
}

inline unsigned long long check_mobility(const unsigned long long P, const unsigned long long O){
	unsigned long long moves, mO, flip1, pre1, flip8, pre8;
	__m128i	PP, mOO, MM, flip, pre;
	mO = O & 0x7e7e7e7e7e7e7e7eULL;
	PP  = _mm_set_epi64x(mirror_v(P), P);
	mOO = _mm_set_epi64x(mirror_v(mO), mO);
	flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 7));				flip1  = mO & (P << 1);		flip8  = O & (P << 8);
	flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 7)));		flip1 |= mO & (flip1 << 1);	flip8 |= O & (flip8 << 8);
	pre  = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 7));				pre1   = mO & (mO << 1);	pre8   = O & (O << 8);
	flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));	flip1 |= pre1 & (flip1 << 2);	flip8 |= pre8 & (flip8 << 16);
	flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));	flip1 |= pre1 & (flip1 << 2);	flip8 |= pre8 & (flip8 << 16);
	MM = _mm_slli_epi64(flip, 7);							moves = flip1 << 1;		moves |= flip8 << 8;
	flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 9));				flip1  = mO & (P >> 1);		flip8  = O & (P >> 8);
	flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 9)));		flip1 |= mO & (flip1 >> 1);	flip8 |= O & (flip8 >> 8);
	pre = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 9));				pre1 >>= 1;			pre8 >>= 8;
	flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));	flip1 |= pre1 & (flip1 >> 2);	flip8 |= pre8 & (flip8 >> 16);
	flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));	flip1 |= pre1 & (flip1 >> 2);	flip8 |= pre8 & (flip8 >> 16);
	MM = _mm_or_si128(MM, _mm_slli_epi64(flip, 9));					moves |= flip1 >> 1;		moves |= flip8 >> 8;
	moves |= _mm_cvtsi128_si64(MM) | mirror_v(_mm_cvtsi128_si64(_mm_unpackhi_epi64(MM, MM)));
	return moves & ~(P | O);
}

inline unsigned long long move(const unsigned long long p, const unsigned long long o, const int& pos){
    __m256i	PP, OO, flip, outflank, mask;
	__m128i	flip2, OP;
	const __m256 exp_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xff800000));
	const __m256i minusone = _mm256_set1_epi64x(-1);

    OP = _mm_set_epi64x(o, p);
	PP = _mm256_broadcastq_epi64(OP);
	OO = _mm256_permute4x64_epi64(_mm256_castsi128_si256(OP), 0x55);

	mask = move_param.rmask_v4[pos].v4;
	outflank = _mm256_andnot_si256(OO, mask);
	outflank = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_cvtepi32_ps(outflank), exp_mask));
	outflank = _mm256_andnot_si256(_mm256_srli_epi32(_mm256_srai_epi32(outflank, 31), 1), outflank);
	outflank = _mm256_and_si256(outflank, _mm256_cmpeq_epi32(_mm256_srli_epi64(outflank, 32), _mm256_setzero_si256()));
	outflank = _mm256_and_si256(outflank, PP);
	flip = _mm256_and_si256(_mm256_sub_epi64(_mm256_setzero_si256(), _mm256_add_epi64(outflank, outflank)), mask);
	mask = move_param.lmask_v4[pos].v4;
	outflank = _mm256_andnot_si256(OO, mask);
	outflank = _mm256_andnot_si256(_mm256_add_epi64(outflank, minusone), outflank);
	outflank = _mm256_and_si256(outflank, PP);
	outflank = _mm256_add_epi64(outflank, minusone);
	outflank = _mm256_add_epi64(outflank, _mm256_srli_epi64(outflank, 63));
	flip = _mm256_or_si256(flip, _mm256_and_si256(outflank, mask));
	flip2 = _mm_or_si128(_mm256_castsi256_si128(flip), _mm256_extracti128_si256(flip, 1));
	flip2 = _mm_or_si128(flip2, _mm_shuffle_epi32(flip2, 0x4e));

    unsigned long long put, rev;
    put = 1ULL << pos;
    rev = _mm_cvtsi128_si64(flip2);
    return p ^ (put | rev);
}

inline int check_confirm(const unsigned long long& grid, const int& idx){
    int i, res = 0;
    for (i = 0; i < hw; ++i){
        if (1 & (grid >> confirm_param.lst[idx][i]))
            res++;
        else
            break;
    }
    return res;
}

inline double evaluate(const unsigned long long p, const unsigned long long o, int canput, int open_val){
    int canput_all = canput;
    double weight_me = 0.0, weight_op = 0.0;
    int me_cnt = 0, op_cnt = 0;
    int confirm_me = 0, confirm_op = 0;
    //int stone_me = 0, stone_op = 0;
    int out_me = 0, out_op = 0;
    unsigned long long mobility, stones;
    int i, j;
    for (i = 0; i < hw2; ++i){
        if (1 & (p >> (hw2 - i - 1))){
            weight_me += eval_param.weight[i];
            me_cnt++;
        } else if (1 & (o >> (hw2 - i - 1))){
            weight_op += eval_param.weight[i];
            op_cnt++;
        }
    }
    mobility = check_mobility(p, o);
    for (i = 0; i < hw2; ++i)
        canput_all += 1 & (mobility >> i);
    stones = p | o;
    for (i = 0; i < hw; i += 2){
        if (stones ^ confirm_param.num[i / 2]){
            for (j = 0; j < 2; ++j){
                confirm_me += max(0, check_confirm(p, i + j) - 1);
                confirm_op += max(0, check_confirm(o, i + j) - 1);
            }
        } else {
            for (j = 1; j < hw - 1; ++j){
                if (1 & (p >> confirm_param.lst[i][j]))
                    confirm_me++;
                else if (1 & (o >> confirm_param.lst[i][j]))
                    confirm_op++;
            }
        }
    }
    confirm_me += 1 & p;
    confirm_me += 1 & (p >> hw_m1);
    confirm_me += 1 & (p >> hw2_mhw);
    confirm_me += 1 & (p >> hw2_m1);
    confirm_op += 1 & o;
    confirm_op += 1 & (o >> hw_m1);
    confirm_op += 1 & (o >> hw2_mhw);
    confirm_op += 1 & (o >> hw2_m1);
    for (i = 0; i < hw2; ++i){
        if (1 & (stones >> i))
            continue;
        out_me += 1 & (p >> (i + 1));
        out_me += 1 & (p >> (i - 1));
        out_me += 1 & (p >> (i + hw));
        out_me += 1 & (p >> (i - hw));
        out_me += 1 & (p >> (i + hw_p1));
        out_me += 1 & (p >> (i - hw_m1));
        out_me += 1 & (p >> (i + hw_m1));
        out_me += 1 & (p >> (i - hw_p1));
        out_op += 1 & (o >> (i + 1));
        out_op += 1 & (o >> (i - 1));
        out_op += 1 & (o >> (i + hw));
        out_op += 1 & (o >> (i - hw));
        out_op += 1 & (o >> (i + hw_p1));
        out_op += 1 & (o >> (i - hw_m1));
        out_op += 1 & (o >> (i + hw_m1));
        out_op += 1 & (o >> (i - hw_p1));
    }
    double weight_proc, canput_proc, confirm_proc, open_proc, out_proc;
    weight_proc = weight_me / max(1, me_cnt) - weight_op / max(1, op_cnt);
    canput_proc = (double)(canput_all - canput) / max(1, canput_all) - (double)canput / max(1, canput_all);
    confirm_proc = (double)confirm_me / max(1, confirm_me + confirm_op) - (double)confirm_op / max(1, confirm_me + confirm_op);
    //stone_proc = 0; //-(double)stone_me / (stone_me + stone_op) + (double)stone_op / (stone_me + stone_op);
    open_proc = 1.0 - eval_param.open_val_threshold * open_val;
    out_proc = -(double)out_me / max(1, out_me + out_op) + (double)out_op / max(1, out_me + out_op);
    return weight_proc * eval_param.weight_weight + 
        canput_proc * eval_param.canput_weight + 
        confirm_proc * eval_param.confirm_weight + 
        open_proc * eval_param.open_weight + 
        out_proc * eval_param.out_weight;
}

inline double end_game(const unsigned long long p, const unsigned long long o){
    int res = 0, i;
    for (i = 0; i < hw2; ++i){
        res += 1 & (p >> i);
        res -= 1 & (o >> i);
    }
    return (double)res * 100000.0;
}

inline int calc_open(unsigned long long stones, unsigned long long rev){
    int i, res = 0;
    for (i = 0; i < hw2; ++i){
        if (1 & (rev >> i)){
            res += 1 - (1 & (stones >> (i + 1)));
            res += 1 - (1 & (stones >> (i - 1)));
            res += 1 - (1 & (stones >> (i + hw)));
            res += 1 - (1 & (stones >> (i - hw)));
            res += 1 - (1 & (stones >> (i + hw_p1)));
            res += 1 - (1 & (stones >> (i - hw_m1)));
            res += 1 - (1 & (stones >> (i + hw_m1)));
            res += 1 - (1 & (stones >> (i - hw_p1)));
        }
    }
    return res;
}

int cmp(grid_priority p, grid_priority q){
    return p.open_val < q.open_val;
}

inline int pop_count_ull(unsigned long long x){
    x = x - ((x >> 1) & 0x5555555555555555);
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
	x = (x * 0x0101010101010101) >> 56;
    return (int)x;
}

double nega_alpha(const unsigned long long p, const unsigned long long o, const int& depth, double alpha, double beta, const int& skip_cnt, const int& canput, int open_val){
    if (skip_cnt == 2)
        return end_game(p, o);
    else if (depth == 0)
        return evaluate(p, o, canput, open_val);
    double val, v, ub, lb;
    int i, n_canput;
    unsigned long long mobility = check_mobility(p, o);
    unsigned long long np, no;
    double priority;
    val = -65.0;
    n_canput = pop_count_ull(mobility);
    if (n_canput == 0)
        return -nega_alpha(o, p, depth, -beta, -alpha, skip_cnt + 1, 0, 0);
    for (i = 0; i < hw2; ++i){
        if (1 & (mobility >> i)){
            np = move(p, o, i);
            no = (np ^ o) & o;
            v = -nega_alpha(no, np, depth - 1, -beta, -alpha, 0, n_canput, calc_open(np | no, np ^ p));
            if (fabs(v) == inf)
                return -inf;
            if (beta <= v)
                return v;
            alpha = max(alpha, v);
        if (val < v)
            val = v;
        }
    }
    return val;
}

double nega_scout(const unsigned long long p, const unsigned long long o, const int& depth, double alpha, double beta, const int& skip_cnt){
    if (search_param.max_depth > search_param.min_max_depth && tim() - search_param.strt > search_param.tl)
        return -inf;
    if (skip_cnt == 2)
        return end_game(p, o);
    double val, v, ub, lb;
    pair<unsigned long long, unsigned long long> grid_all;
    grid_all.first = p;
    grid_all.second = o;
    lb = search_param.memo_lb[grid_all];
    if (lb != 0.0){
        if (lb >= beta)
            return lb;
        alpha = max(alpha, lb);
    }
    ub = search_param.memo_ub[grid_all];
    if (ub != 0.0){
        if (alpha >= ub || ub == lb)
            return ub;
        beta = min(beta, ub);
    }
    int i, n_canput = 0, open_val;
    unsigned long long mobility = check_mobility(p, o);
    unsigned long long np, no;
    double priority;
    vector<grid_priority> lst;
    for (i = 0; i < hw2; ++i){
        if (1 & (mobility >> i)){
            ++n_canput;
            np = move(p, o, i);
            no = (np ^ o) & o;
            grid_priority tmp;
            tmp.open_val = calc_open(np | no, np ^ p);
            tmp.p = np;
            tmp.o = no;
            lst.push_back(tmp);
        }
    }
    if (n_canput == 0)
        return -nega_scout(o, p, depth, -beta, -alpha, skip_cnt + 1);
    if (n_canput > 1)
        sort(lst.begin(), lst.end(), cmp);
    if (depth > simple_threshold)
        v = -nega_scout(lst[0].o, lst[0].p, depth - 1, -beta, -alpha, 0);
    else
        v = -nega_alpha(lst[0].o, lst[0].p, depth - 1, -beta, -alpha, 0, n_canput, lst[0].open_val);
    val = v;
    if (fabs(v) == inf)
        return -inf;
    if (beta <= v)
        return v;
    alpha = max(alpha, v);
    for (i = 1; i < n_canput; ++i){
        if (depth > simple_threshold)
            v = -nega_scout(lst[i].o, lst[i].p, depth - 1, -alpha - window, -alpha, 0);
        else
            v = -nega_alpha(lst[i].o, lst[i].p, depth - 1, -alpha - window, -alpha, 0, n_canput, lst[i].open_val);
        if (fabs(v) == inf)
            return -inf;
        if (beta <= v)
            return v;
        if (alpha < v){
            alpha = v;
            if (depth > simple_threshold)
                v = -nega_scout(lst[i].o, lst[i].p, depth - 1, -beta, -alpha, 0);
            else
                v = -nega_alpha(lst[i].o, lst[i].p, depth - 1, -beta, -alpha, 0, n_canput, lst[i].open_val);
            if (fabs(v) == inf)
                return -inf;
            if (beta <= v)
                return v;
            alpha = max(alpha, v);
        }
        if (val < v)
            val = v;
    }
    if (val <= alpha)
        search_param.memo_ub[grid_all] = val;
    else if (val >= beta)
        search_param.memo_lb[grid_all] = val;
    else {
        search_param.memo_ub[grid_all] = val;
        search_param.memo_lb[grid_all] = val;
    }
    return val;
}

double map_double(double s, double e, double x){
    return s + (e - s) * x;
}

int cmp_main(grid_priority_main p, grid_priority_main q){
    return p.priority > q.priority;
}

int main(int argc, char* argv[]){
    int ansy, ansx, outy, outx, i, canput, former_depth = 7, former_vacant = hw2 - 4;
    double score, max_score;
    double weight_weight_s, canput_weight_s, confirm_weight_s, stone_weight_s, open_weight_s, out_weight_s, weight_weight_e, canput_weight_e, confirm_weight_e, stone_weight_e, open_weight_e, out_weight_e;
    unsigned long long in_mobility;
    unsigned long long p, o, np, no;
    vector<grid_priority_main> lst;
    pair<unsigned long long, unsigned long long> grid_all;
    int elem;
    int action_count;
    double game_ratio;
    int vacant_cnt, ai_player;
    
    init(argc, argv);
    cin >> ai_player;
    cin >> search_param.tl;
    
    if (ai_player == 0){
        cerr << "AI initialized AI is Black" << endl;
    }else{
        cerr << "AI initialized AI is White" << endl;
    }
    while (true){
        outy = -1;
        outx = -1;
        vacant_cnt = 0;
        p = 0;
        o = 0;
        in_mobility = 0;
        canput = 0;
        for (i = 0; i < hw2; ++i){
            cin >> elem;
            vacant_cnt += (int)(elem == -1 || elem == 2);
            in_mobility <<= 1;
            in_mobility += (int)(elem == 2);
            canput += (int)(elem == 2);
            p <<= 1;
            o <<= 1;
            p += (int)(elem == ai_player);
            o += (int)(elem == 1 - ai_player);
        }
        
        if (vacant_cnt > 14)
            search_param.min_max_depth = max(5, former_depth + vacant_cnt - former_vacant);
        else
            search_param.min_max_depth = 15;
        
        //search_param.min_max_depth = 2;
        cerr << "start depth " << search_param.min_max_depth << endl;
        search_param.max_depth = search_param.min_max_depth;
        former_vacant = vacant_cnt;
        lst.clear();
        for (i = 0; i < hw2; ++i){
            if (1 & (in_mobility >> i)){
                np = move(p, o, i);
                no = (np ^ o) & o;
                grid_priority_main tmp;
                tmp.open_val = calc_open(np | no, np ^ p);
                tmp.priority = -tmp.open_val;
                tmp.p = np;
                tmp.o = no;
                tmp.move = i;
                lst.push_back(tmp);
            }
        }
        if (canput > 1)
            sort(lst.begin(), lst.end(), cmp_main);
        outy = -1;
        outx = -1;
        search_param.strt = tim();
        while (tim() - search_param.strt < search_param.tl / 2){
            search_param.memo_ub.clear();
            search_param.memo_lb.clear();
            game_ratio = (double)(hw2 - vacant_cnt + search_param.max_depth) / hw2;
            eval_param.weight_weight = map_double(eval_param.weight_se[0], eval_param.weight_se[6], game_ratio);
            eval_param.canput_weight = map_double(eval_param.weight_se[1], eval_param.weight_se[7], game_ratio);
            eval_param.confirm_weight = map_double(eval_param.weight_se[2], eval_param.weight_se[8], game_ratio);
            eval_param.open_weight = map_double(eval_param.weight_se[3], eval_param.weight_se[9], game_ratio);
            eval_param.out_weight = map_double(eval_param.weight_se[4], eval_param.weight_se[10], game_ratio);
            eval_param.open_val_threshold = map_double(eval_param.weight_se[5], eval_param.weight_se[11], game_ratio);
            for (i = 0; i < hw2; i++)
                eval_param.weight[i] = map_double(eval_param.weight_s[i], eval_param.weight_e[i], game_ratio);
            max_score = -6500000.0;
            for (i = 0; i < canput; ++i){
                score = -nega_scout(lst[i].o, lst[i].p, search_param.max_depth - 1, -6500000.0, -max_score, 0);
                if (fabs(score) == inf){
                    max_score = -inf;
                    break;
                }
                lst[i].priority = score;
                if (score > max_score){
                    max_score = score;
                    ansy = (hw2 - lst[i].move - 1) / hw;
                    ansx = (hw2 - lst[i].move - 1) % hw;
                }
            }
            if (max_score == -inf){
                cerr << "depth " << search_param.max_depth << " timeoout" << endl;
                break;
            }
            former_depth = search_param.max_depth;
            outy = ansy;
            outx = ansx;
            if (canput > 1)
                sort(lst.begin(), lst.end(), cmp_main);
            cerr << "depth " << search_param.max_depth;
            for (i = 0; i < 1; ++i){
                cerr << "  " << ((hw2 - lst[i].move - 1) / hw) << ((hw2 - lst[i].move - 1) % hw) << " " << lst[i].priority;
            }
            cerr << " time " << tim() - search_param.strt << endl;
            if (vacant_cnt < search_param.max_depth || fabs(max_score) >= 100000.0){
                cerr << "game end" << endl;
                break;
            }
            search_param.max_depth++;
        }
        cout << outy << " " << outx << endl;
    }
    return 0;
}