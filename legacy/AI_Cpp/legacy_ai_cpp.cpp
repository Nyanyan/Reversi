#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx")

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
#include <immintrin.h>

using namespace std;

#define hw 8
#define hw2 64
#define window 0.00001
#define simple_threshold 2

const int dy[8] = {0, 1, 0, -1, 1, 1, -1, -1};
const int dx[8] = {1, 0, -1, 0, 1, -1, 1, -1};

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


const double weight_s[hw2] = {
    3.2323232323232323, 0.23088023088023088, 1.3852813852813852, 1.0389610389610389, 1.0389610389610389, 1.3852813852813852, 0.23088023088023088, 3.2323232323232323,
    0.23088023088023088, 0.0, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.0, 0.23088023088023088,
    1.3852813852813852, 0.9004329004329005, 1.0389610389610389, 0.9466089466089466, 0.9466089466089466, 1.0389610389610389, 0.9004329004329005, 1.3852813852813852,
    1.0389610389610389, 0.9004329004329005, 0.9466089466089466, 0.9235209235209235, 0.9235209235209235, 0.9466089466089466, 0.9004329004329005, 1.0389610389610389,
    1.0389610389610389, 0.9004329004329005, 0.9466089466089466, 0.9235209235209235, 0.9235209235209235, 0.9466089466089466, 0.9004329004329005, 1.0389610389610389,
    1.3852813852813852, 0.9004329004329005, 1.0389610389610389, 0.9466089466089466, 0.9466089466089466, 1.0389610389610389, 0.9004329004329005, 1.3852813852813852,
    0.23088023088023088, 0.0, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.0, 0.23088023088023088,
    3.2323232323232323, 0.23088023088023088, 1.3852813852813852, 1.0389610389610389, 1.0389610389610389, 1.3852813852813852, 0.23088023088023088, 3.2323232323232323
};

const double weight_e[hw2] = {
    2.2153846153846155, 0.36923076923076925, 1.6, 0.9846153846153847, 0.9846153846153847, 1.6, 0.36923076923076925, 2.2153846153846155, 
    0.36923076923076925, 0.0, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.0, 0.36923076923076925,
    1.6, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 1.6,
    0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847,
    0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847,
    1.6, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 1.6,
    0.36923076923076925, 0.0, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 0.0, 0.36923076923076925,
    2.2153846153846155, 0.36923076923076925, 1.6, 0.9846153846153847, 0.9846153846153847, 1.6, 0.36923076923076925, 2.2153846153846155
};

double weight[hw2];

/*
const double weight[hw2] = {
        3.739463601532567, 0.3065134099616858, 1.2260536398467432, 0.9195402298850575, 0.9195402298850575, 1.2260536398467432, 0.3065134099616858, 3.739463601532567,   
        0.3065134099616858, 0.0, 0.9195402298850575, 0.7969348659003831, 0.7969348659003831, 0.9195402298850575, 0.0, 0.3065134099616858,
        1.2260536398467432, 0.9195402298850575, 1.103448275862069, 0.9195402298850575, 0.9195402298850575, 1.103448275862069, 0.9195402298850575, 1.2260536398467432,   
        0.9195402298850575, 0.7969348659003831, 0.9195402298850575, 0.9808429118773946, 0.9808429118773946, 0.9195402298850575, 0.7969348659003831, 0.9195402298850575, 
        0.9195402298850575, 0.7969348659003831, 0.9195402298850575, 0.9808429118773946, 0.9808429118773946, 0.9195402298850575, 0.7969348659003831, 0.9195402298850575,
        1.2260536398467432, 0.9195402298850575, 1.103448275862069, 0.9195402298850575, 0.9195402298850575, 1.103448275862069, 0.9195402298850575, 1.2260536398467432,
        0.3065134099616858, 0.0, 0.9195402298850575, 0.7969348659003831, 0.7969348659003831, 0.9195402298850575, 0.0, 0.3065134099616858,
        3.739463601532567, 0.3065134099616858, 1.2260536398467432, 0.9195402298850575, 0.9195402298850575, 1.2260536398467432, 0.3065134099616858, 3.739463601532567
};
*/
/*
const double weight[hw2] = {
        3.4877384196185286, 0.4359673024523161, 1.3079019073569482, 0.9809264305177112, 0.9809264305177112, 1.3079019073569482, 0.4359673024523161, 3.4877384196185286, 
        0.4359673024523161, 0.0, 0.7629427792915532, 0.7629427792915532, 0.7629427792915532, 0.7629427792915532, 0.0, 0.4359673024523161,
        1.3079019073569482, 0.7629427792915532, 1.1989100817438691, 0.9373297002724795, 0.9373297002724795, 1.1989100817438691, 0.7629427792915532, 1.3079019073569482, 
        0.9809264305177112, 0.7629427792915532, 0.9373297002724795, 0.9373297002724795, 0.9373297002724795, 0.9373297002724795, 0.7629427792915532, 0.9809264305177112, 
        0.9809264305177112, 0.7629427792915532, 0.9373297002724795, 0.9373297002724795, 0.9373297002724795, 0.9373297002724795, 0.7629427792915532, 0.9809264305177112,
        1.3079019073569482, 0.7629427792915532, 1.1989100817438691, 0.9373297002724795, 0.9373297002724795, 1.1989100817438691, 0.7629427792915532, 1.3079019073569482,
        0.4359673024523161, 0.0, 0.7629427792915532, 0.7629427792915532, 0.7629427792915532, 0.7629427792915532, 0.0, 0.4359673024523161,
        3.4877384196185286, 0.4359673024523161, 1.3079019073569482, 0.9809264305177112, 0.9809264305177112, 1.3079019073569482, 0.4359673024523161, 3.4877384196185286
};
*/


const int confirm_lst[hw][hw] = {
    {63, 62, 61, 60, 59, 58, 57, 56},
    {56, 57, 58, 59, 60, 61, 62, 63},
    {63, 55, 47, 39, 31, 23, 15,  7},
    { 7, 15, 23, 31, 39, 47, 55, 63},
    { 7,  6,  5,  4,  3,  2,  1,  0},
    { 0,  1,  2,  3,  4,  5,  6,  7},
    {56, 48, 40 , 32, 24, 16, 8,  0},
    { 0,  8, 16, 24, 32, 40, 48, 56}
};

const unsigned long long confirm_num[4] = {
    0b0000000000000000000000000000000000000000000000000000000011111111,
    0b0000000100000001000000010000000100000001000000010000000100000001,
    0b1111111100000000000000000000000000000000000000000000000000000000,
    0b1000000010000000100000001000000010000000100000001000000010000000
};

struct grid_priority{
    unsigned long long me;
    unsigned long long op;
    int open_val;
};

struct grid_priority_main{
    double priority;
    unsigned long long me;
    unsigned long long op;
    int move;
};

int ai_player;
double weight_weight, canput_weight, confirm_weight, stone_weight, open_weight, out_weight;
int max_depth, vacant_cnt;
double game_ratio;
unordered_map<pair<unsigned long long, unsigned long long>, double, HashPair> memo1, memo2; 
unordered_map<pair<unsigned long long, unsigned long long>, double, HashPair> memo_lb, memo_ub;
unsigned long long marked;
int min_max_depth;
int strt, tl;

#ifdef _MSC_VER
	#define	mirror_v(x)	_byteswap_uint64(x)
#else
	#define	mirror_v(x)	__builtin_bswap64(x)
#endif

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

typedef union {
	unsigned long long	ull[4];
	#ifdef __AVX2__
		__m256i	v4;
	#endif
	__m128i	v2[2];
} V4DI;

static const V4DI lmask_v4[hw2] = {
	{{ 0x00000000000000fe, 0x0101010101010100, 0x8040201008040200, 0x0000000000000000 }},
	{{ 0x00000000000000fc, 0x0202020202020200, 0x0080402010080400, 0x0000000000000100 }},
	{{ 0x00000000000000f8, 0x0404040404040400, 0x0000804020100800, 0x0000000000010200 }},
	{{ 0x00000000000000f0, 0x0808080808080800, 0x0000008040201000, 0x0000000001020400 }},
	{{ 0x00000000000000e0, 0x1010101010101000, 0x0000000080402000, 0x0000000102040800 }},
	{{ 0x00000000000000c0, 0x2020202020202000, 0x0000000000804000, 0x0000010204081000 }},
	{{ 0x0000000000000080, 0x4040404040404000, 0x0000000000008000, 0x0001020408102000 }},
	{{ 0x0000000000000000, 0x8080808080808000, 0x0000000000000000, 0x0102040810204000 }},
	{{ 0x000000000000fe00, 0x0101010101010000, 0x4020100804020000, 0x0000000000000000 }},
	{{ 0x000000000000fc00, 0x0202020202020000, 0x8040201008040000, 0x0000000000010000 }},
	{{ 0x000000000000f800, 0x0404040404040000, 0x0080402010080000, 0x0000000001020000 }},
	{{ 0x000000000000f000, 0x0808080808080000, 0x0000804020100000, 0x0000000102040000 }},
	{{ 0x000000000000e000, 0x1010101010100000, 0x0000008040200000, 0x0000010204080000 }},
	{{ 0x000000000000c000, 0x2020202020200000, 0x0000000080400000, 0x0001020408100000 }},
	{{ 0x0000000000008000, 0x4040404040400000, 0x0000000000800000, 0x0102040810200000 }},
	{{ 0x0000000000000000, 0x8080808080800000, 0x0000000000000000, 0x0204081020400000 }},
	{{ 0x0000000000fe0000, 0x0101010101000000, 0x2010080402000000, 0x0000000000000000 }},
	{{ 0x0000000000fc0000, 0x0202020202000000, 0x4020100804000000, 0x0000000001000000 }},
	{{ 0x0000000000f80000, 0x0404040404000000, 0x8040201008000000, 0x0000000102000000 }},
	{{ 0x0000000000f00000, 0x0808080808000000, 0x0080402010000000, 0x0000010204000000 }},
	{{ 0x0000000000e00000, 0x1010101010000000, 0x0000804020000000, 0x0001020408000000 }},
	{{ 0x0000000000c00000, 0x2020202020000000, 0x0000008040000000, 0x0102040810000000 }},
	{{ 0x0000000000800000, 0x4040404040000000, 0x0000000080000000, 0x0204081020000000 }},
	{{ 0x0000000000000000, 0x8080808080000000, 0x0000000000000000, 0x0408102040000000 }},
	{{ 0x00000000fe000000, 0x0101010100000000, 0x1008040200000000, 0x0000000000000000 }},
	{{ 0x00000000fc000000, 0x0202020200000000, 0x2010080400000000, 0x0000000100000000 }},
	{{ 0x00000000f8000000, 0x0404040400000000, 0x4020100800000000, 0x0000010200000000 }},
	{{ 0x00000000f0000000, 0x0808080800000000, 0x8040201000000000, 0x0001020400000000 }},
	{{ 0x00000000e0000000, 0x1010101000000000, 0x0080402000000000, 0x0102040800000000 }},
	{{ 0x00000000c0000000, 0x2020202000000000, 0x0000804000000000, 0x0204081000000000 }},
	{{ 0x0000000080000000, 0x4040404000000000, 0x0000008000000000, 0x0408102000000000 }},
	{{ 0x0000000000000000, 0x8080808000000000, 0x0000000000000000, 0x0810204000000000 }},
	{{ 0x000000fe00000000, 0x0101010000000000, 0x0804020000000000, 0x0000000000000000 }},
	{{ 0x000000fc00000000, 0x0202020000000000, 0x1008040000000000, 0x0000010000000000 }},
	{{ 0x000000f800000000, 0x0404040000000000, 0x2010080000000000, 0x0001020000000000 }},
	{{ 0x000000f000000000, 0x0808080000000000, 0x4020100000000000, 0x0102040000000000 }},
	{{ 0x000000e000000000, 0x1010100000000000, 0x8040200000000000, 0x0204080000000000 }},
	{{ 0x000000c000000000, 0x2020200000000000, 0x0080400000000000, 0x0408100000000000 }},
	{{ 0x0000008000000000, 0x4040400000000000, 0x0000800000000000, 0x0810200000000000 }},
	{{ 0x0000000000000000, 0x8080800000000000, 0x0000000000000000, 0x1020400000000000 }},
	{{ 0x0000fe0000000000, 0x0101000000000000, 0x0402000000000000, 0x0000000000000000 }},
	{{ 0x0000fc0000000000, 0x0202000000000000, 0x0804000000000000, 0x0001000000000000 }},
	{{ 0x0000f80000000000, 0x0404000000000000, 0x1008000000000000, 0x0102000000000000 }},
	{{ 0x0000f00000000000, 0x0808000000000000, 0x2010000000000000, 0x0204000000000000 }},
	{{ 0x0000e00000000000, 0x1010000000000000, 0x4020000000000000, 0x0408000000000000 }},
	{{ 0x0000c00000000000, 0x2020000000000000, 0x8040000000000000, 0x0810000000000000 }},
	{{ 0x0000800000000000, 0x4040000000000000, 0x0080000000000000, 0x1020000000000000 }},
	{{ 0x0000000000000000, 0x8080000000000000, 0x0000000000000000, 0x2040000000000000 }},
	{{ 0x00fe000000000000, 0x0100000000000000, 0x0200000000000000, 0x0000000000000000 }},
	{{ 0x00fc000000000000, 0x0200000000000000, 0x0400000000000000, 0x0100000000000000 }},
	{{ 0x00f8000000000000, 0x0400000000000000, 0x0800000000000000, 0x0200000000000000 }},
	{{ 0x00f0000000000000, 0x0800000000000000, 0x1000000000000000, 0x0400000000000000 }},
	{{ 0x00e0000000000000, 0x1000000000000000, 0x2000000000000000, 0x0800000000000000 }},
	{{ 0x00c0000000000000, 0x2000000000000000, 0x4000000000000000, 0x1000000000000000 }},
	{{ 0x0080000000000000, 0x4000000000000000, 0x8000000000000000, 0x2000000000000000 }},
	{{ 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, 0x4000000000000000 }},
	{{ 0xfe00000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0xfc00000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0xf800000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0xf000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0xe000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0xc000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }}
};

static const V4DI rmask_v4[hw2] = {
	{{ 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x0000000000000003, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x0000000000000007, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x000000000000000f, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x000000000000001f, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x000000000000003f, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x000000000000007f, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }},
	{{ 0x0000000000000000, 0x0000000000000001, 0x0000000000000000, 0x0000000000000002 }},
	{{ 0x0000000000000100, 0x0000000000000002, 0x0000000000000001, 0x0000000000000004 }},
	{{ 0x0000000000000300, 0x0000000000000004, 0x0000000000000002, 0x0000000000000008 }},
	{{ 0x0000000000000700, 0x0000000000000008, 0x0000000000000004, 0x0000000000000010 }},
	{{ 0x0000000000000f00, 0x0000000000000010, 0x0000000000000008, 0x0000000000000020 }},
	{{ 0x0000000000001f00, 0x0000000000000020, 0x0000000000000010, 0x0000000000000040 }},
	{{ 0x0000000000003f00, 0x0000000000000040, 0x0000000000000020, 0x0000000000000080 }},
	{{ 0x0000000000007f00, 0x0000000000000080, 0x0000000000000040, 0x0000000000000000 }},
	{{ 0x0000000000000000, 0x0000000000000101, 0x0000000000000000, 0x0000000000000204 }},
	{{ 0x0000000000010000, 0x0000000000000202, 0x0000000000000100, 0x0000000000000408 }},
	{{ 0x0000000000030000, 0x0000000000000404, 0x0000000000000201, 0x0000000000000810 }},
	{{ 0x0000000000070000, 0x0000000000000808, 0x0000000000000402, 0x0000000000001020 }},
	{{ 0x00000000000f0000, 0x0000000000001010, 0x0000000000000804, 0x0000000000002040 }},
	{{ 0x00000000001f0000, 0x0000000000002020, 0x0000000000001008, 0x0000000000004080 }},
	{{ 0x00000000003f0000, 0x0000000000004040, 0x0000000000002010, 0x0000000000008000 }},
	{{ 0x00000000007f0000, 0x0000000000008080, 0x0000000000004020, 0x0000000000000000 }},
	{{ 0x0000000000000000, 0x0000000000010101, 0x0000000000000000, 0x0000000000020408 }},
	{{ 0x0000000001000000, 0x0000000000020202, 0x0000000000010000, 0x0000000000040810 }},
	{{ 0x0000000003000000, 0x0000000000040404, 0x0000000000020100, 0x0000000000081020 }},
	{{ 0x0000000007000000, 0x0000000000080808, 0x0000000000040201, 0x0000000000102040 }},
	{{ 0x000000000f000000, 0x0000000000101010, 0x0000000000080402, 0x0000000000204080 }},
	{{ 0x000000001f000000, 0x0000000000202020, 0x0000000000100804, 0x0000000000408000 }},
	{{ 0x000000003f000000, 0x0000000000404040, 0x0000000000201008, 0x0000000000800000 }},
	{{ 0x000000007f000000, 0x0000000000808080, 0x0000000000402010, 0x0000000000000000 }},
	{{ 0x0000000000000000, 0x0000000001010101, 0x0000000000000000, 0x0000000002040810 }},
	{{ 0x0000000100000000, 0x0000000002020202, 0x0000000001000000, 0x0000000004081020 }},
	{{ 0x0000000300000000, 0x0000000004040404, 0x0000000002010000, 0x0000000008102040 }},
	{{ 0x0000000700000000, 0x0000000008080808, 0x0000000004020100, 0x0000000010204080 }},
	{{ 0x0000000f00000000, 0x0000000010101010, 0x0000000008040201, 0x0000000020408000 }},
	{{ 0x0000001f00000000, 0x0000000020202020, 0x0000000010080402, 0x0000000040800000 }},
	{{ 0x0000003f00000000, 0x0000000040404040, 0x0000000020100804, 0x0000000080000000 }},
	{{ 0x0000007f00000000, 0x0000000080808080, 0x0000000040201008, 0x0000000000000000 }},
	{{ 0x0000000000000000, 0x0000000101010101, 0x0000000000000000, 0x0000000204081020 }},
	{{ 0x0000010000000000, 0x0000000202020202, 0x0000000100000000, 0x0000000408102040 }},
	{{ 0x0000030000000000, 0x0000000404040404, 0x0000000201000000, 0x0000000810204080 }},
	{{ 0x0000070000000000, 0x0000000808080808, 0x0000000402010000, 0x0000001020408000 }},
	{{ 0x00000f0000000000, 0x0000001010101010, 0x0000000804020100, 0x0000002040800000 }},
	{{ 0x00001f0000000000, 0x0000002020202020, 0x0000001008040201, 0x0000004080000000 }},
	{{ 0x00003f0000000000, 0x0000004040404040, 0x0000002010080402, 0x0000008000000000 }},
	{{ 0x00007f0000000000, 0x0000008080808080, 0x0000004020100804, 0x0000000000000000 }},
	{{ 0x0000000000000000, 0x0000010101010101, 0x0000000000000000, 0x0000020408102040 }},
	{{ 0x0001000000000000, 0x0000020202020202, 0x0000010000000000, 0x0000040810204080 }},
	{{ 0x0003000000000000, 0x0000040404040404, 0x0000020100000000, 0x0000081020408000 }},
	{{ 0x0007000000000000, 0x0000080808080808, 0x0000040201000000, 0x0000102040800000 }},
	{{ 0x000f000000000000, 0x0000101010101010, 0x0000080402010000, 0x0000204080000000 }},
	{{ 0x001f000000000000, 0x0000202020202020, 0x0000100804020100, 0x0000408000000000 }},
	{{ 0x003f000000000000, 0x0000404040404040, 0x0000201008040201, 0x0000800000000000 }},
	{{ 0x007f000000000000, 0x0000808080808080, 0x0000402010080402, 0x0000000000000000 }},
	{{ 0x0000000000000000, 0x0001010101010101, 0x0000000000000000, 0x0002040810204080 }},
	{{ 0x0100000000000000, 0x0002020202020202, 0x0001000000000000, 0x0004081020408000 }},
	{{ 0x0300000000000000, 0x0004040404040404, 0x0002010000000000, 0x0008102040800000 }},
	{{ 0x0700000000000000, 0x0008080808080808, 0x0004020100000000, 0x0010204080000000 }},
	{{ 0x0f00000000000000, 0x0010101010101010, 0x0008040201000000, 0x0020408000000000 }},
	{{ 0x1f00000000000000, 0x0020202020202020, 0x0010080402010000, 0x0040800000000000 }},
	{{ 0x3f00000000000000, 0x0040404040404040, 0x0020100804020100, 0x0080000000000000 }},
	{{ 0x7f00000000000000, 0x0080808080808080, 0x0040201008040201, 0x0000000000000000 }}
};

inline unsigned long long move(const unsigned long long grid_me, const unsigned long long grid_op, const int& pos){
    __m256i	PP, OO, flip, outflank, mask;
	__m128i	flip2, OP;
	const __m256 exp_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xff800000));
	const __m256i minusone = _mm256_set1_epi64x(-1);

    OP = _mm_set_epi64x(grid_op, grid_me);
	PP = _mm256_broadcastq_epi64(OP);
	OO = _mm256_permute4x64_epi64(_mm256_castsi128_si256(OP), 0x55);

	mask = rmask_v4[pos].v4;
	outflank = _mm256_andnot_si256(OO, mask);
	outflank = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_cvtepi32_ps(outflank), exp_mask));
	outflank = _mm256_andnot_si256(_mm256_srli_epi32(_mm256_srai_epi32(outflank, 31), 1), outflank);
	outflank = _mm256_and_si256(outflank, _mm256_cmpeq_epi32(_mm256_srli_epi64(outflank, 32), _mm256_setzero_si256()));
	outflank = _mm256_and_si256(outflank, PP);
	flip = _mm256_and_si256(_mm256_sub_epi64(_mm256_setzero_si256(), _mm256_add_epi64(outflank, outflank)), mask);
	mask = lmask_v4[pos].v4;
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
    return grid_me ^ (put | rev);
}

inline int check_confirm(const unsigned long long& grid, const int& idx){
    int i, res = 0;
    for (i = 0; i < hw; ++i){
        if (1 & (grid >> confirm_lst[idx][i]))
            res++;
        else
            break;
    }
    return res;
}

inline double evaluate(unsigned long long grid_me, unsigned long long grid_op, int canput, int open_val){
    int canput_all = canput;
    double weight_me = 0.0, weight_op = 0.0;
    int me_cnt = 0, op_cnt = 0;
    int confirm_me = 0, confirm_op = 0;
    //int stone_me = 0, stone_op = 0;
    int out_me = 0, out_op = 0;
    unsigned long long mobility, stones;
    int i, j;
    for (i = 0; i < hw2; ++i){
        if (1 & (grid_me >> (hw2 - i - 1))){
            weight_me += weight[i];
            me_cnt++;
        } else if (1 & (grid_op >> (hw2 - i - 1))){
            weight_op += weight[i];
            op_cnt++;
        }
    }
    mobility = check_mobility(grid_me, grid_op);
    for (i = 0; i < hw2; ++i)
        canput_all += 1 & (mobility >> i);
    stones = grid_me | grid_op;
    for (i = 0; i < hw; i += 2){
        if (stones ^ confirm_num[i / 2]){
            for (j = 0; j < 2; ++j){
                confirm_me += max(0, check_confirm(grid_me, i + j) - 1);
                confirm_op += max(0, check_confirm(grid_op, i + j) - 1);
            }
        } else {
            for (j = 1; j < hw - 1; ++j){
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
    /*
    for (i = 0; i < hw2; ++i){
        stone_me += 1 & (grid_me >> i);
        stone_op += 1 & (grid_op >> i);
    }
    */
    for (i = 0; i < hw2; ++i){
        if (1 & (stones >> i))
            continue;
        out_me += 1 & (grid_me >> (i + 1));
        out_me += 1 & (grid_me >> (i - 1));
        out_me += 1 & (grid_me >> (i + hw));
        out_me += 1 & (grid_me >> (i - hw));
        out_me += 1 & (grid_me >> (i + hw + 1));
        out_me += 1 & (grid_me >> (i - hw + 1));
        out_me += 1 & (grid_me >> (i + hw - 1));
        out_me += 1 & (grid_me >> (i - hw - 1));
        out_op += 1 & (grid_op >> (i + 1));
        out_op += 1 & (grid_op >> (i - 1));
        out_op += 1 & (grid_op >> (i + hw));
        out_op += 1 & (grid_op >> (i - hw));
        out_op += 1 & (grid_op >> (i + hw + 1));
        out_op += 1 & (grid_op >> (i - hw + 1));
        out_op += 1 & (grid_op >> (i + hw - 1));
        out_op += 1 & (grid_op >> (i - hw - 1));
    }
    double weight_proc, canput_proc, confirm_proc, open_proc, out_proc;
    weight_proc = weight_me / me_cnt - weight_op / op_cnt;
    canput_proc = (double)(canput_all - canput) / max(1, canput_all) - (double)canput / max(1, canput_all);
    confirm_proc = (double)confirm_me / max(1, confirm_me + confirm_op) - (double)confirm_op / max(1, confirm_me + confirm_op);
    //stone_proc = 0; //-(double)stone_me / (stone_me + stone_op) + (double)stone_op / (stone_me + stone_op);
    open_proc = max(-1.0, (double)(3 - open_val) / 3);
    out_proc = -(double)out_me / max(1, out_me + out_op) + (double)out_op / max(1, out_me + out_op);
    return max(-0.999, min(0.999, weight_proc * weight_weight + canput_proc * canput_weight + confirm_proc * confirm_weight + open_proc * open_weight + out_proc * out_weight));
}

inline double end_game(unsigned long long grid_me, unsigned long long grid_op){
    int res = 0, i;
    for (i = 0; i < hw2; ++i){
        res += 1 & (grid_me >> i);
        res -= 1 & (grid_op >> i);
    }
    return (double)res;
}

inline int calc_open(unsigned long long stones, unsigned long long rev){
    int i, res = 0;
    for (i = 0; i < hw2; ++i){
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
    return p.open_val < q.open_val;
}

inline int pop_count_ull(unsigned long long x){
    x = x - ((x >> 1) & 0x5555555555555555);
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
	x = (x * 0x0101010101010101) >> 56;
    return (int)x;
}

double nega_alpha(const unsigned long long& grid_me, const unsigned long long& grid_op, const int& depth, double alpha, double beta, const int& skip_cnt, const int& canput, int open_val){
    if (skip_cnt == 2)
        return end_game(grid_me, grid_op);
    else if (depth == 0)
        return evaluate(grid_me, grid_op, canput, open_val);
    double val, v, ub, lb;
    int i, n_canput;
    unsigned long long mobility = check_mobility(grid_me, grid_op);
    unsigned long long n_grid_me, n_grid_op, x;
    double priority;
    val = -65.0;
    n_canput = pop_count_ull(mobility);
    if (n_canput == 0)
        return -nega_alpha(grid_op, grid_me, depth, -beta, -alpha, skip_cnt + 1, 0, 0);
    for (i = 0; i < hw2; ++i){
        if (1 & (mobility >> i)){
            n_grid_me = move(grid_me, grid_op, i);
            n_grid_op = (n_grid_me ^ grid_op) & grid_op;
            v = -nega_alpha(n_grid_op, n_grid_me, depth - 1, -beta, -alpha, 0, n_canput, calc_open(n_grid_me | n_grid_op, n_grid_me ^ grid_me));
            if (fabs(v) == 100000000.0)
                return -100000000.0;
            if (beta <= v)
                return v;
            alpha = max(alpha, v);
        if (val < v)
            val = v;
        }
    }
    return val;
}

double nega_scout(const unsigned long long& grid_me, const unsigned long long& grid_op, const int& depth, double alpha, double beta, const int& skip_cnt){
    if (max_depth > min_max_depth && tim() - strt > tl)
        return -100000000.0;
    if (skip_cnt == 2)
        return end_game(grid_me, grid_op);
    double val, v, ub, lb;
    pair<unsigned long long, unsigned long long> grid_all;
    grid_all.first = grid_me;
    grid_all.second = grid_op;
    lb = memo_lb[grid_all];
    if (lb != 0.0){
        if (lb >= beta)
            return lb;
        alpha = max(alpha, lb);
    }
    ub = memo_ub[grid_all];
    if (ub != 0.0){
        if (alpha >= ub || ub == lb)
            return ub;
        beta = min(beta, ub);
    }
    int i, n_canput = 0, open_val;
    unsigned long long mobility = check_mobility(grid_me, grid_op);
    unsigned long long n_grid_me, n_grid_op;
    double priority;
    vector<grid_priority> lst;
    for (i = 0; i < hw2; ++i){
        if (1 & (mobility >> i)){
            n_canput++;
            n_grid_me = move(grid_me, grid_op, i);
            n_grid_op = (n_grid_me ^ grid_op) & grid_op;
            grid_priority tmp;
            tmp.open_val = calc_open(n_grid_me | n_grid_op, n_grid_me ^ grid_me);
            tmp.me = n_grid_me;
            tmp.op = n_grid_op;
            lst.push_back(tmp);
        }
    }
    if (n_canput == 0)
        return -nega_scout(grid_op, grid_me, depth, -beta, -alpha, skip_cnt + 1);
    if (n_canput > 1)
        sort(lst.begin(), lst.end(), cmp);
    if (depth > simple_threshold)
        v = -nega_scout(lst[0].op, lst[0].me, depth - 1, -beta, -alpha, 0);
    else
        v = -nega_alpha(lst[0].op, lst[0].me, depth - 1, -beta, -alpha, 0, n_canput, lst[0].open_val);
    val = v;
    if (fabs(v) == 100000000.0)
        return -100000000.0;
    if (beta <= v)
        return v;
    alpha = max(alpha, v);
    for (i = 1; i < n_canput; ++i){
        if (depth > simple_threshold)
            v = -nega_scout(lst[i].op, lst[i].me, depth - 1, -alpha - window, -alpha, 0);
        else
            v = -nega_alpha(lst[i].op, lst[i].me, depth - 1, -alpha - window, -alpha, 0, n_canput, lst[i].open_val);
        if (fabs(v) == 100000000.0)
            return -100000000.0;
        if (beta <= v)
            return v;
        if (alpha < v){
            alpha = v;
            if (depth > simple_threshold)
                v = -nega_scout(lst[i].op, lst[i].me, depth - 1, -beta, -alpha, 0);
            else
                v = -nega_alpha(lst[i].op, lst[i].me, depth - 1, -beta, -alpha, 0, n_canput, lst[i].open_val);
            if (fabs(v) == 100000000.0)
                return -100000000.0;
            if (beta <= v)
                return v;
            alpha = max(alpha, v);
        }
        if (val < v)
            val = v;
    }
    if (val <= alpha)
        memo_ub[grid_all] = val;
    else if (val >= beta)
        memo_lb[grid_all] = val;
    else {
        memo_ub[grid_all] = val;
        memo_lb[grid_all] = val;
    }
    return val;
}

double map_double(double s, double e, double x){
    return s + (e - s) * x;
}

int cmp_main(grid_priority_main p, grid_priority_main q){
    return p.priority > q.priority;
}

int main(){
    int ansy, ansx, outy, outx, i, canput, former_depth = 7, former_vacant = hw2 - 4;
    double score, max_score;
    double weight_weight_s, canput_weight_s, confirm_weight_s, stone_weight_s, open_weight_s, out_weight_s, weight_weight_e, canput_weight_e, confirm_weight_e, stone_weight_e, open_weight_e, out_weight_e;
    unsigned long long in_grid_me, in_grid_op, in_mobility, grid_me, grid_op;
    vector<grid_priority_main> lst;
    pair<unsigned long long, unsigned long long> grid_all;
    int elem;
    int action_count;
    cin >> ai_player;
    cin >> tl;
    weight_weight_s = 0.4;
    canput_weight_s = 0.45;
    confirm_weight_s = 0.0;
    open_weight_s = 0.1;
    out_weight_s = 0.05;
    
    weight_weight_e = 0.1;
    canput_weight_e = 0.55;
    confirm_weight_e = 0.3;
    open_weight_e = 0.1;
    out_weight_e = -0.05;
    
    if (ai_player == 0){
        cerr << "AI initialized AI is Black" << endl;
    }else{
        cerr << "AI initialized AI is White" << endl;
    }
    while (true){
        outy = -1;
        outx = -1;
        vacant_cnt = 0;
        in_grid_me = 0;
        in_grid_op = 0;
        in_mobility = 0;
        canput = 0;
        for (i = 0; i < hw2; ++i){
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
        if (vacant_cnt > 14)
            min_max_depth = max(5, former_depth + vacant_cnt - former_vacant);
        else
            min_max_depth = 15;
        //min_max_depth = 2;
        cerr << "start depth " << min_max_depth << endl;
        max_depth = min_max_depth;
        former_vacant = vacant_cnt;
        lst.clear();
        for (i = 0; i < hw2; ++i){
            if (1 & (in_mobility >> i)){
                grid_me = move(in_grid_me, in_grid_op, i);
                grid_op = (grid_me ^ in_grid_op) & in_grid_op;
                grid_all.first = grid_me;
                grid_all.second = grid_op;
                grid_priority_main tmp;
                tmp.priority = -0.1 * calc_open(grid_me | grid_op, grid_me ^ in_grid_me);
                tmp.me = grid_me;
                tmp.op = grid_op;
                tmp.move = i;
                lst.push_back(tmp);
            }
        }
        if (canput > 1)
            sort(lst.begin(), lst.end(), cmp_main);
        strt = tim();
        while (tim() - strt < tl / 2){
            memo_ub.clear();
            memo_lb.clear();
            game_ratio = (double)(hw2 - vacant_cnt + max_depth) / hw2;
            weight_weight = map_double(weight_weight_s, weight_weight_e, game_ratio);
            canput_weight = map_double(canput_weight_s, canput_weight_e, game_ratio);
            confirm_weight = map_double(confirm_weight_s, confirm_weight_e, game_ratio);
            //stone_weight = map_double(stone_weight_s, stone_weight_e, game_ratio);
            open_weight = map_double(open_weight_s, open_weight_e, game_ratio);
            out_weight = map_double(out_weight_s, out_weight_e, game_ratio);
            for (i = 0; i < hw2; i++)
                weight[i] = map_double(weight_s[i], weight_e[i], game_ratio);
            max_score = -65.0;
            for (i = 0; i < canput; ++i){
                score = -nega_scout(lst[i].op, lst[i].me, max_depth - 1, -65.0, -max_score, 0);
                if (fabs(score) == 100000000.0){
                    max_score = -100000000.0;
                    break;
                }
                lst[i].priority = score;
                if (score > max_score){
                    max_score = score;
                    ansy = (hw2 - lst[i].move - 1) / hw;
                    ansx = (hw2 - lst[i].move - 1) % hw;
                }
            }
            if (max_score == -100000000.0){
                cerr << "depth " << max_depth << " timeoout" << endl;
                break;
            }
            former_depth = max_depth;
            outy = ansy;
            outx = ansx;
            if (canput > 1)
                sort(lst.begin(), lst.end(), cmp_main);
            cerr << "depth " << max_depth;
            for (i = 0; i < 1; ++i){
                cerr << "  " << ((hw2 - lst[i].move - 1) / hw) << ((hw2 - lst[i].move - 1) % hw) << " " << lst[i].priority;
            }
            cerr << " time " << tim() - strt << endl;
            if (vacant_cnt < max_depth || fabs(max_score) >= 1.0){
                cerr << "game end" << endl;
                break;
            }
            max_depth++;
            //break;
        }
        cout << outy << " " << outx << endl;
    }
    return 0;
}