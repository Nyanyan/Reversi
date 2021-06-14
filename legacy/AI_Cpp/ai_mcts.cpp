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
#include <queue>

using namespace std;

int xorx=123456789, xory=362436069, xorz=521288629, xorw=88675123;
double myrandom(){
    int t = (xorx^(xorx<<11));
    xorx = xory;
    xory = xorz;
    xorz = xorw;
    xorw = xorw=(xorw^(xorw>>19))^(t^(t>>8));
    return (double)(xorw) / 2147483648.0;
}
int randint(int fr, int to){
    return fr + (int)(myrandom() * (to - fr + 1));
}

#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw2_m1 63
#define hw2_mhw 56
#define window 0.00001
#define simple_threshold 2
#define inf 1000000000000.0
#define pattern_num 6
#define bias (1.0 * 1.41421356)
#define expand_threshold 10

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
    double weight_weight, canput_weight, confirm_weight, stone_weight, open_weight, out_weight, pattern_weight;
    double weight_se[14];
    double open_val_threshold;
    double weight_pat_s[pattern_num];
    double weight_pat_e[pattern_num];
    double weight_pat[pattern_num];
    unsigned long long pat_mask_h_p[pattern_num], pat_mask_h_o[pattern_num], pat_mask_h_p_m[pattern_num], pat_mask_h_o_m[pattern_num], pat_mask_v_p[pattern_num], pat_mask_v_o[pattern_num], pat_mask_v_p_m[pattern_num], pat_mask_v_o_m[pattern_num];
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

struct grid_eval{
    int n;
    int m;
    double val;
    bool seen;
};

struct grid_node{
    unsigned long long p, o;
    double priority;
};

struct search_param{
    unordered_map<pair<unsigned long long, unsigned long long>, grid_eval, HashPair> win_rate;
    int playout_cnt;
    int strt, tl;
};

struct grid_priority{
    double priority;
    unsigned long long p;
    unsigned long long o;
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

unsigned long long transpose(unsigned long long x) {
    unsigned long long t;
    const unsigned long long k1 = 0xaa00aa00aa00aa00;
    const unsigned long long k2 = 0xcccc0000cccc0000;
    const unsigned long long k4 = 0xf0f0f0f00f0f0f0f;
    t  =       x ^ (x << 36) ;
    x ^= k4 & (t ^ (x >> 36));
    t  = k2 & (x ^ (x << 18));
    x ^=       t ^ (t >> 18) ;
    t  = k1 & (x ^ (x <<  9));
    x ^=       t ^ (t >>  9) ;
    return x;
}

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
    const char* file;
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
    for (i = 0; i < 14; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.weight_se[i] = atof(cbuf);
    }
    for (i = 0; i < pattern_num; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.weight_pat_s[i] = atof(cbuf);
    }
    for (i = 0; i < pattern_num; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.weight_pat_e[i] = atof(cbuf);
    }
    fclose(fp);
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
    for (i = 0; i < pattern_num; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("const.txt broken");
            exit(1);
        }
        eval_param.pat_mask_h_p[i] = stoull(cbuf);
    }
    for (i = 0; i < pattern_num; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("const.txt broken");
            exit(1);
        }
        eval_param.pat_mask_h_o[i] = stoull(cbuf);
    }
    fclose(fp);
    for (i = 0; i < pattern_num; i++){
        eval_param.pat_mask_h_p_m[i] = mirror_v(eval_param.pat_mask_h_p[i]);
        eval_param.pat_mask_h_o_m[i] = mirror_v(eval_param.pat_mask_h_o[i]);
        eval_param.pat_mask_v_p[i] = transpose(eval_param.pat_mask_h_p[i]);
        eval_param.pat_mask_v_o[i] = transpose(eval_param.pat_mask_h_o[i]);
        eval_param.pat_mask_v_p_m[i] = transpose(eval_param.pat_mask_h_p_m[i]);
        eval_param.pat_mask_v_o_m[i] = transpose(eval_param.pat_mask_h_o_m[i]);
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
    int out_me = 0, out_op = 0;
    double pattern_me = 0.0, pattern_op = 0.0;
    unsigned long long mobility, stones;
    unsigned long long p1, p2, o1, o2;
    int i, j;
    p1 = p << hw2_mhw;
    p2 = p >> hw_m1;
    o1 = o << hw2_mhw;
    o2 = o >> hw_m1;
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
    confirm_me += 1 & p2;
    confirm_me += 1 & (p >> hw2_mhw);
    confirm_me += 1 & (p >> hw2_m1);
    confirm_op += 1 & o;
    confirm_op += 1 & o2;
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
    for (i = 0; i < pattern_num; i++){
        if ((p & eval_param.pat_mask_h_p[i]) == eval_param.pat_mask_h_p[i] && (o & eval_param.pat_mask_h_o[i]) == eval_param.pat_mask_h_o[i])
            pattern_me += eval_param.weight_pat[i];
        if ((p & eval_param.pat_mask_h_p_m[i]) == eval_param.pat_mask_h_p_m[i] && (o & eval_param.pat_mask_h_o_m[i]) == eval_param.pat_mask_h_o_m[i])
            pattern_me += eval_param.weight_pat[i];
        if ((p & eval_param.pat_mask_v_p[i]) == eval_param.pat_mask_v_p[i] && (o & eval_param.pat_mask_v_o[i]) == eval_param.pat_mask_v_o[i])
            pattern_me += eval_param.weight_pat[i];
        if ((p & eval_param.pat_mask_v_p_m[i]) == eval_param.pat_mask_v_p_m[i] && (o & eval_param.pat_mask_v_o_m[i]) == eval_param.pat_mask_v_o_m[i])
            pattern_me += eval_param.weight_pat[i];
        if ((p1 & eval_param.pat_mask_h_p[i]) == eval_param.pat_mask_h_p[i] && (o1 & eval_param.pat_mask_h_o[i]) == eval_param.pat_mask_h_o[i])
            pattern_me += eval_param.weight_pat[i];
        if ((p1 & eval_param.pat_mask_h_p_m[i]) == eval_param.pat_mask_h_p_m[i] && (o1 & eval_param.pat_mask_h_o_m[i]) == eval_param.pat_mask_h_o_m[i])
            pattern_me += eval_param.weight_pat[i];
        if ((p2 & eval_param.pat_mask_v_p[i]) == eval_param.pat_mask_v_p[i] && (o2 & eval_param.pat_mask_v_o[i]) == eval_param.pat_mask_v_o[i])
            pattern_me += eval_param.weight_pat[i];
        if ((p2 & eval_param.pat_mask_v_p_m[i]) == eval_param.pat_mask_v_p_m[i] && (o2 & eval_param.pat_mask_v_o_m[i]) == eval_param.pat_mask_v_o_m[i])
            pattern_me += eval_param.weight_pat[i];
        if ((o & eval_param.pat_mask_h_p[i]) == eval_param.pat_mask_h_p[i] && (p & eval_param.pat_mask_h_o[i]) == eval_param.pat_mask_h_o[i])
            pattern_op += eval_param.weight_pat[i];
        if ((o & eval_param.pat_mask_h_p_m[i]) == eval_param.pat_mask_h_p_m[i] && (p & eval_param.pat_mask_h_o_m[i]) == eval_param.pat_mask_h_o_m[i])
            pattern_op += eval_param.weight_pat[i];
        if ((o & eval_param.pat_mask_v_p[i]) == eval_param.pat_mask_v_p[i] && (p & eval_param.pat_mask_v_o[i]) == eval_param.pat_mask_v_o[i])
            pattern_op += eval_param.weight_pat[i];
        if ((o & eval_param.pat_mask_v_p_m[i]) == eval_param.pat_mask_v_p_m[i] && (p & eval_param.pat_mask_v_o_m[i]) == eval_param.pat_mask_v_o_m[i])
            pattern_op += eval_param.weight_pat[i];
        if ((o1 & eval_param.pat_mask_h_p[i]) == eval_param.pat_mask_h_p[i] && (p1 & eval_param.pat_mask_h_o[i]) == eval_param.pat_mask_h_o[i])
            pattern_op += eval_param.weight_pat[i];
        if ((o1 & eval_param.pat_mask_h_p_m[i]) == eval_param.pat_mask_h_p_m[i] && (p1 & eval_param.pat_mask_h_o_m[i]) == eval_param.pat_mask_h_o_m[i])
            pattern_op += eval_param.weight_pat[i];
        if ((o2 & eval_param.pat_mask_v_p[i]) == eval_param.pat_mask_v_p[i] && (p2 & eval_param.pat_mask_v_o[i]) == eval_param.pat_mask_v_o[i])
            pattern_op += eval_param.weight_pat[i];
        if ((o2 & eval_param.pat_mask_v_p_m[i]) == eval_param.pat_mask_v_p_m[i] && (p2 & eval_param.pat_mask_v_o_m[i]) == eval_param.pat_mask_v_o_m[i])
            pattern_op += eval_param.weight_pat[i];
    }
    double weight_proc, canput_proc, confirm_proc, open_proc, out_proc, pattern_proc;
    weight_proc = weight_me / max(1, me_cnt) - weight_op / max(1, op_cnt);
    canput_proc = (double)(canput_all - canput) / max(1, canput_all) - (double)canput / max(1, canput_all);
    confirm_proc = (double)confirm_me / max(1, confirm_me + confirm_op) - (double)confirm_op / max(1, confirm_me + confirm_op);
    open_proc = 1.0 - eval_param.open_val_threshold * open_val;
    out_proc = -(double)out_me / max(1, out_me + out_op) + (double)out_op / max(1, out_me + out_op);
    pattern_proc = pattern_me - pattern_op;
    return weight_proc * eval_param.weight_weight + 
        canput_proc * eval_param.canput_weight + 
        confirm_proc * eval_param.confirm_weight + 
        open_proc * eval_param.open_weight + 
        out_proc * eval_param.out_weight +
        pattern_proc * eval_param.pattern_weight;
}

inline int end_game(const unsigned long long p, const unsigned long long o){
    int res = 0, i;
    for (i = 0; i < hw2; ++i){
        res += 1 & (p >> i);
        res -= 1 & (o >> i);
    }
    //return (double)res * 1000.0;
    if (res > 0)
        return 1;
    else if (res == 0)
        return 0;
    else
        return -1;
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
    return p.priority > q.priority;
}

inline int pop_count_ull(unsigned long long x){
    x = x - ((x >> 1) & 0x5555555555555555);
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
	x = (x * 0x0101010101010101) >> 56;
    return (int)x;
}

void expand(const unsigned long long p, const unsigned long long o, const unsigned long long mobility){
    unsigned long long np, no;
    int i;
    pair<unsigned long long, unsigned long long> grid;
    for (i = 0; i < hw2; ++i){
        if (1 & (mobility >> i)){
            np = move(p, o, i);
            no = (np ^ o) & o;
            grid.first = no;
            grid.second = np;
            search_param.win_rate[grid].seen = true;
            search_param.win_rate[grid].n = 0;
            search_param.win_rate[grid].m = 0;
            //search_param.win_rate[grid].val = -evaluate(no, np, n_canput, calc_open(np | no, np ^ p));
        }
    }
}

int playout(const unsigned long long p, const unsigned long long o, const int skip_cnt){
    if (skip_cnt == 2)
        return -end_game(p, o);
    unsigned long long mobility = check_mobility(p, o);
    int n_canput = pop_count_ull(mobility);
    if (n_canput == 0)
        return -playout(o, p, skip_cnt + 1);
    unsigned long long np, no;
    int i;
    vector<grid_priority> lst;
    for (i = 0; i < hw2; ++i){
        if (1 & (mobility >> i)){
            np = move(p, o, i);
            no = (np ^ o) & o;
            grid_priority tmp;
            tmp.p = np;
            tmp.o = no;
            tmp.priority = -evaluate(no, np, n_canput, calc_open(np | no, np ^ p)) + myrandom() * 0.2 - 0.1;
            lst.push_back(tmp);
        }
    }
    double max_score = -inf;
    for (i = 0; i < n_canput; i++){
        if (lst[i].priority > max_score){
            max_score = lst[i].priority;
            no = lst[i].o;
            np = lst[i].p;
        }
    }
    return -playout(no, np, 0);
}

int mcts(const unsigned long long p, const unsigned long long o, const int skip_cnt){
    pair<unsigned long long, unsigned long long> grid_now;
    int res;
    grid_now.first = p;
    grid_now.second = o;
    ++search_param.win_rate[grid_now].n;
    if (skip_cnt == 2){
        res = -end_game(p, o);
        search_param.win_rate[grid_now].m += res;
        return res;
    }
    pair<unsigned long long, unsigned long long> grid_new;
    unsigned long long mobility = check_mobility(p, o);
    int n_canput = pop_count_ull(mobility);
    if (n_canput == 0){
        res = -playout(o, p, skip_cnt + 1);
        search_param.win_rate[grid_now].m += res;
        return res;
    }
    if (search_param.win_rate[grid_now].n == expand_threshold)
        expand(p, o, mobility);
    unsigned long long np, no;
    vector<grid_priority> lst;
    int i;
    for (i = 0; i < hw2; ++i){
        if (1 & (mobility >> i)){
            np = move(p, o, i);
            no = (np ^ o) & o;
            grid_new.first = no;
            grid_new.second = np;
            grid_priority tmp;
            tmp.p = np;
            tmp.o = no;
            if (search_param.win_rate[grid_new].seen){
                if (search_param.win_rate[grid_new].n > 0)
                    tmp.priority = (double)search_param.win_rate[grid_new].m / search_param.win_rate[grid_new].n + bias * sqrt(log((double)search_param.win_rate[grid_now].n) / (double)search_param.win_rate[grid_new].n); //bias * search_param.win_rate[grid].val;
                else
                    tmp.priority = inf;
            } else
                tmp.priority = myrandom();
            lst.push_back(tmp);
        }
    }
    double max_priority = -inf;
    for (i = 0; i < n_canput; ++i){
        if (lst[i].priority > max_priority){
            max_priority = lst[i].priority;
            grid_new.first = lst[i].o;
            grid_new.second = lst[i].p;
        }
    }
    if (search_param.win_rate[grid_new].seen)
        res = -mcts(grid_new.first, grid_new.second, 0);
    else
        res = -playout(grid_new.first, grid_new.second, 0);
    search_param.win_rate[grid_now].m += res;
    return res;
}

double map_double(double s, double e, double x){
    return s + (e - s) * x;
}

int main(int argc, char* argv[]){
    int ansy, ansx, outy, outx, i, canput, former_depth = 7, former_vacant = hw2 - 4;
    double score, max_score;
    double weight_weight_s, canput_weight_s, confirm_weight_s, stone_weight_s, open_weight_s, out_weight_s, weight_weight_e, canput_weight_e, confirm_weight_e, stone_weight_e, open_weight_e, out_weight_e;
    unsigned long long in_mobility;
    unsigned long long p, o, np, no;
    pair<unsigned long long, unsigned long long> grid;
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
        former_vacant = vacant_cnt;
        outy = -1;
        outx = -1;
        search_param.strt = tim();
        game_ratio = (double)(hw2 - vacant_cnt) / hw2;
        eval_param.weight_weight = map_double(eval_param.weight_se[0], eval_param.weight_se[1], game_ratio);
        eval_param.canput_weight = map_double(eval_param.weight_se[2], eval_param.weight_se[3], game_ratio);
        eval_param.confirm_weight = map_double(eval_param.weight_se[4], eval_param.weight_se[5], game_ratio);
        eval_param.open_weight = map_double(eval_param.weight_se[6], eval_param.weight_se[7], game_ratio);
        eval_param.out_weight = map_double(eval_param.weight_se[8], eval_param.weight_se[9], game_ratio);
        eval_param.open_val_threshold = map_double(eval_param.weight_se[10], eval_param.weight_se[11], game_ratio);
        eval_param.pattern_weight = map_double(eval_param.weight_se[12], eval_param.weight_se[13], game_ratio);
        for (i = 0; i < hw2; i++)
            eval_param.weight[i] = map_double(eval_param.weight_s[i], eval_param.weight_e[i], game_ratio);
        search_param.playout_cnt = 0;
        search_param.win_rate.clear();
        search_param.win_rate[make_pair(p, o)].seen = true;
        expand(p, o, in_mobility);
        while (tim() - search_param.strt < search_param.tl){
            ++search_param.playout_cnt;
            mcts(p, o, 0);
        }
        max_score = -inf;
        for (i = 0; i < hw2; i++){
            if (1 & (in_mobility >> i)){
                np = move(p, o, i);
                no = (np ^ o) & o;
                grid.first = no;
                grid.second = np;
                score = (double)search_param.win_rate[grid].m / max(1, search_param.win_rate[grid].n);
                //cerr << search_param.win_rate[grid].n << " " << score << "  ";
                cerr << search_param.win_rate[grid].n << " ";
                if (score > max_score){
                    max_score = score;
                    ansy = (hw2 - i - 1) / hw;
                    ansx = (hw2 - i - 1) % hw;
                }
            }
        }
        cerr << endl;
        outy = ansy;
        outx = ansx;
        cerr << ansy << ansx << " " << max_score << " " << search_param.playout_cnt << " time " << tim() - search_param.strt << endl;
        cout << outy << " " << outx << endl;
    }
    return 0;
}