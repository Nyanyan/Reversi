#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx")

// Reversi AI C++ version 3

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
//#include <bitset>

using namespace std;

#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw2_m1 63
#define hw2_mhw 56
#define window 0.00001
#define simple_threshold 3
#define inf 100000000.0
#define param_num 12
#define board_index_num 38
#define pattern_num 10

struct hash_arr{
    //static size_t m_hash_arr_random;
    size_t operator()(const int *p) const {
        size_t seed = 0;
        seed ^= (size_t)p[0];
        seed ^= (size_t)p[1] << 7;
        seed ^= (size_t)p[2] << 14;
        seed ^= (size_t)p[3] << 21;
        seed ^= (size_t)p[4] << 28;
        seed ^= (size_t)p[5] << 35;
        seed ^= (size_t)p[6] << 42;
        seed ^= (size_t)p[7] << 49;
        //seed ^= m_hash_arr_random + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

//size_t hash_arr::m_hash_arr_random = (size_t) random_device()();

struct board_param{
    unsigned long long trans[board_index_num][6561][hw];
    bool legal[6561][hw];
    int put[hw2][board_index_num];
    int board_translate[board_index_num][8];
    int board_rev_translate[hw2][4][2];
    int pattern_space[board_index_num];
    int reverse[6561];
    int pow3[10];
    int rev_bit3[6561][8];
};

struct eval_param{
    double weight_s[hw2], weight_m[hw2], weight_e[hw2];
    double weight[hw2];
    double pattern_weight, cnt_weight, canput_weight;
    double cnt_bias;
    double weight_sme[param_num];
    double avg_canput[hw2];
    /*= {
        0.00, 0.00, 0.00, 0.00, 4.00, 3.00, 4.00, 2.00,
        9.00, 5.00, 6.00, 6.00, 5.00, 8.38, 5.69, 9.13,
        5.45, 6.98, 6.66, 9.38, 6.98, 9.29, 7.29, 9.32, 
        7.37, 9.94, 7.14, 9.78, 7.31, 10.95, 7.18, 9.78, 
        7.76, 9.21, 7.33, 8.81, 7.20, 8.48, 7.23, 8.00, 
        6.92, 7.57, 6.62, 7.13, 6.38, 6.54, 5.96, 6.18, 
        5.62, 5.64, 5.18, 5.18, 4.60, 4.48, 4.06, 3.67, 
        3.39, 3.11, 2.66, 2.30, 1.98, 1.53, 1.78, 0.67
    };
    */
    double pattern_space[pattern_num];
    double pattern_data[pattern_num][6561];
    int pattern_same[board_index_num];
    int canput[6561];
    int cnt_p[6561], cnt_o[6561];
};

struct search_param{
    unordered_map<int*, double, hash_arr> memo_lb, memo_ub;
    int max_depth;
    int min_max_depth;
    int strt, tl;
    int turn;
    int searched_nodes;
};

struct board_priority_move{
    int b[board_index_num];
    double priority;
    int move;
};

struct board_priority{
    int b[board_index_num];
    double priority;
};

board_param board_param;
eval_param eval_param;
search_param search_param;

inline int pop_count_ull(unsigned long long x){
    x = x - ((x >> 1) & 0x5555555555555555);
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
	x = (x * 0x0101010101010101) >> 56;
    return (int)x;
}

int tim(){
    return static_cast<int>(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count());
}

void print_board(int* board){
    int i, j, idx, tmp;
    for (i = 0; i < hw; ++i){
        tmp = board[i];
        for (j = 0; j < hw; ++j){
            if (tmp % 3 == 0){
                cerr << ". ";
            }else if (tmp % 3 == 1){
                cerr << "P ";
            }else{
                cerr << "O ";
            }
            tmp /= 3;
        }
        cerr << endl;
    }
    cerr << endl;
}

int reverse_line(int a) {
    int res = 0;
    for (int i = 0; i < hw; ++i) {
        res <<= 1;
        res |= 1 & (a >> i);
    }
    return res;
}

inline unsigned long long check_mobility(const int p, const int o){
	int p1 = p << 1;
    int res = ~(p1 | o) & (p1 + o);
    int p_rev = reverse_line(p), o_rev = reverse_line(o);
    int p2 = p_rev << 1;
    res |= reverse_line(~(p2 | o_rev) & (p2 + o_rev));
    res &= ~(p | o);
    // cerr << bitset<8>(p) << " " << bitset<8>(o) << " " << bitset<8>(res) << endl;
    return res;
}

int trans(int pt, int k) {
    if (k == 0)
        return pt >> 1;
    else
        return pt << 1;
}

int move_line(int p, int o, const int place) {
    int rev = 0;
    int rev2, mask, tmp;
    int pt = 1 << place;
    for (int k = 0; k < 2; ++k) {
        rev2 = 0;
        mask = trans(pt, k);
        while (mask && (mask & o)) {
            rev2 |= mask;
            tmp = mask;
            mask = trans(tmp, k);
            if (mask & p)
                rev |= rev2;
        }
    }
    // cerr << bitset<8>(p) << " " << bitset<8>(o) << " " << bitset<8>(rev | pt) << endl;
    return rev | pt;
}

int create_p(int idx){
    int res = 0;
    for (int i = 0; i < hw; ++i){
        if (idx % 3 == 1){
            res |= 1 << i;
        }
        idx /= 3;
    }
    return res;
}

int create_o(int idx){
    int res = 0;
    for (int i = 0; i < hw; ++i){
        if (idx % 3 == 2){
            res |= 1 << i;
        }
        idx /= 3;
    }
    return res;
}

int board_reverse(int idx){
    int p = create_p(idx);
    int o = create_o(idx);
    int res = 0;
    for (int i = hw_m1; i >= 0; --i){
        res *= 3;
        if (1 & (p >> i))
            res += 2;
        else if (1 & (o >> i))
            ++res;
    }
    return res;
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
    int i, j, k, l;
    for (i = 0; i < 10; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        weight_buf[i] = atof(cbuf);
    }
    for (i = 0; i < hw2; ++i)
        eval_param.weight_s[i] = weight_buf[translate[i]];
    for (i = 0; i < 10; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        weight_buf[i] = atof(cbuf);
    }
    for (i = 0; i < hw2; ++i)
        eval_param.weight_m[i] = weight_buf[translate[i]];
    for (i = 0; i < 10; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        weight_buf[i] = atof(cbuf);
    }
    for (i = 0; i < hw2; ++i)
        eval_param.weight_e[i] = weight_buf[translate[i]];
    for (i = 0; i < param_num; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.weight_sme[i] = atof(cbuf);
    }
    fclose(fp);
    if ((fp = fopen("const.txt", "r")) == NULL){
        printf("const.txt not exist");
        exit(1);
    }
    for (i = 0; i < hw2; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("const.txt broken");
            exit(1);
        }
        eval_param.avg_canput[i] = atof(cbuf) * 1.2;
    }
    for (i = 0; i < board_index_num; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("const.txt broken");
            exit(1);
        }
        board_param.pattern_space[i] = atoi(cbuf);
    }
    for (i = 0; i < board_index_num; ++i){
        for (j = 0; j < board_param.pattern_space[i]; ++j){
            if (!fgets(cbuf, 1024, fp)){
                printf("const.txt broken");
                exit(1);
            }
            board_param.board_translate[i][j] = atoi(cbuf);
        }
    }
    for (i = 0; i < board_index_num; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("const.txt broken");
            exit(1);
        }
        eval_param.pattern_same[i] = atoi(cbuf);
    }
    fclose(fp);
    int idx;
    for (i = 0; i < hw2; ++i){
        idx = 0;
        for (j = 0; j < board_index_num; ++j){
            for (k = 0; k < board_param.pattern_space[j]; ++k){
                if (board_param.board_translate[j][k] == i){
                    board_param.board_rev_translate[i][idx][0] = j;
                    board_param.board_rev_translate[i][idx++][1] = k;
                }
            }
        }
        for (j = idx; j < 4; ++j)
            board_param.board_rev_translate[i][j][0] = -1;
    }
    for (i = 0; i < hw2; ++i){
        for (j = 0; j < board_index_num; ++j){
            board_param.put[i][j] = -1;
            for (k = 0; k < board_param.pattern_space[j]; ++k){
                if (board_param.board_translate[j][k] == i)
                    board_param.put[i][j] = k;
            }
        }
    }
    for (i = 0; i < board_index_num; ++i)
        eval_param.pattern_space[eval_param.pattern_same[i]] = board_param.pattern_space[i];
    eval_param.pattern_space[10] = 8;
    if ((fp = fopen("param_pattern.txt", "r")) == NULL){
        printf("param_pattern.txt not exist");
        exit(1);
    }
    for (i = 0; i < pattern_num; ++i){
        for (j = 0; j < (int)pow(3, eval_param.pattern_space[i]); ++j){
            if (!fgets(cbuf, 1024, fp)){
                printf("param_pattern.txt broken");
                exit(1);
            }
            eval_param.pattern_data[i][j] = atof(cbuf);
        }
    }
    fclose(fp);
    int p, o, mobility, canput_num, rev;
    for (i = 0; i < 6561; ++i){
        board_param.reverse[i] = board_reverse(i);
        p = reverse_line(create_p(i));
        o = reverse_line(create_o(i));
        eval_param.cnt_p[i] = 0;
        eval_param.cnt_o[i] = 0;
        for (j = 0; j < hw; ++j){
            eval_param.cnt_p[i] += 1 & (p >> j);
            eval_param.cnt_o[i] += 1 & (o >> j);
        }
        mobility = check_mobility(p, o);
        canput_num = 0;
        for (j = 0; j < hw; ++j){
            if (1 & (mobility >> (hw_m1 - j))){
                rev = move_line(p, o, hw_m1 - j);
                ++canput_num;
                board_param.legal[i][j] = true;
                for (k = 0; k < board_index_num; ++k){
                    board_param.trans[k][i][j] = 0;
                    for (l = 0; l < board_param.pattern_space[k]; ++l)
                        board_param.trans[k][i][j] |= (unsigned long long)(1 & (rev >> (7 - l))) << board_param.board_translate[k][l];
                }
            } else
                board_param.legal[i][j] = false;
        }
        eval_param.canput[i] = canput_num;
    }
    for (i = 0; i < 10; ++i)
        board_param.pow3[i] = (int)pow(3, i);
    for (i = 0; i < 6561; ++i){
        for (j = 0; j < 8; ++j)
            board_param.rev_bit3[i][j] = board_param.pow3[j] * (2 - (i / board_param.pow3[j]) % 3);
    }
}

inline double pattern_eval(const int *board){
    int i;
    double res = 0.0, tmp_res;
    for (i = 0; i < board_index_num; ++i)
        res += eval_param.pattern_data[eval_param.pattern_same[i]][board[i]];
    return res;
}

inline double canput_eval(const int *board){
    int i;
    int res = 0;
    for (i = 0; i < board_index_num; ++i)
        res += eval_param.canput[board[i]];
    return ((double)res - eval_param.avg_canput[search_param.turn]) / ((double)res + eval_param.avg_canput[search_param.turn]);
}

inline double cnt_eval(const int *board){
    int i;
    int res_p = 0.0, res_o = 0.0;
    for (i = 0; i < pattern_num; ++i){
        res_p += eval_param.cnt_p[board[i]];
        res_o += eval_param.cnt_o[board[i]];
    }
    return (double)(res_p * eval_param.cnt_bias - res_o) / (res_p * eval_param.cnt_bias + res_o);
}

inline double evaluate(const int *board){
    double pattern = pattern_eval(board);
    double cnt = cnt_eval(board);
    double canput = canput_eval(board);
    return 
        pattern * eval_param.pattern_weight + 
        cnt * eval_param.cnt_weight + 
        canput * eval_param.canput_weight;
}

inline double end_game(const int *board){
    int res = 0, i, j, p, o;
    for (i = 0; i < hw; ++i){
        res += eval_param.cnt_p[board[i]];
        res -= eval_param.cnt_o[board[i]];
    }
    return (double)res * 1000.0;
}

void move(int *board, int (&res)[board_index_num], int coord){
    int i, j, tmp;
    unsigned long long rev = 0;
    for (i = 0; i < board_index_num; ++i){
        res[i] = board_param.reverse[board[i]];
        if (board_param.put[coord][i] != -1)
            rev |= board_param.trans[i][board[i]][board_param.put[coord][i]];
    }
    for (i = 0; i < hw2; ++i){
        if (1 & (rev >> i)){
            for (j = 0; j < 4; ++j){
                if (board_param.board_rev_translate[i][j][0] == -1)
                    break;
                res[board_param.board_rev_translate[i][j][0]] += board_param.rev_bit3[res[board_param.board_rev_translate[i][j][0]]][board_param.board_rev_translate[i][j][1]];
                //res[board_param.board_rev_translate[i][j][0]] += board_param.pow3[board_param.board_rev_translate[i][j][1]] * (2 - (res[board_param.board_rev_translate[i][j][0]] / board_param.pow3[board_param.board_rev_translate[i][j][1]]) % 3);
            }
        }
    }
}

int cmp(board_priority p, board_priority q){
    return p.priority > q.priority;
}

double nega_alpha(int *board, const int& depth, double alpha, double beta, const int& skip_cnt){
    ++search_param.searched_nodes;
    if (skip_cnt == 2)
        return end_game(board);
    else if (depth == 0){
        return evaluate(board);
    }
    bool is_pass = true;
    int i, j, k, put;
    double val = -inf, v;
    int n_board[board_index_num];
    for (j = 0; j < hw2; ++j){
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[j][i] != -1){
                if (board_param.legal[board[i]][board_param.put[j][i]]){
                    is_pass = false;
                    put = i * hw + j;
                    move(board, n_board, j);
                    v = -nega_alpha(n_board, depth - 1, -beta, -alpha, 0);
                    if (fabs(v) == inf)
                        return -inf;
                    if (beta <= v)
                        return v;
                    alpha = max(alpha, v);
                    if (val < v)
                        val = v;
                    break;
                }
            }
        }
    }
    if (is_pass){
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_alpha(n_board, depth - 1, -beta, -alpha, skip_cnt + 1);
    }
    return val;
}

double nega_scout(int *board, const int& depth, double alpha, double beta, const int& skip_cnt){
    ++search_param.searched_nodes;
    if (search_param.max_depth > search_param.min_max_depth && tim() - search_param.strt > search_param.tl)
        return -inf;
    if (skip_cnt == 2)
        return end_game(board);
    double ub, lb;
    lb = search_param.memo_lb[board];
    if (lb != 0.0){
        if (lb >= beta)
            return lb;
        alpha = max(alpha, lb);
    }
    ub = search_param.memo_ub[board];
    if (ub != 0.0){
        if (alpha >= ub || ub == lb)
            return ub;
        beta = min(beta, ub);
    }
    if (alpha >= beta)
        return alpha;
    int i, j, k, canput = 0;
    double val = -inf, v;
    vector<board_priority> lst;
    for (j = 0; j < hw2; ++j){
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[j][i] != -1){
                if (board_param.legal[board[i]][board_param.put[j][i]]){
                    ++canput;
                    int n_board[board_index_num];
                    board_priority tmp;
                    move(board, tmp.b, j);
                    tmp.priority = evaluate(tmp.b);
                    lst.push_back(tmp);
                    break;
                }
            }
        }
    }
    if (canput == 0){
        int n_board[board_index_num];
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_scout(n_board, depth - 1, -beta, -alpha, skip_cnt + 1);
    }
    if (canput > 1)
        sort(lst.begin(), lst.end(), cmp);
    if (depth > simple_threshold)
        v = -nega_scout(lst[0].b, depth - 1, -beta, -alpha, 0);
    else
        v = -nega_alpha(lst[0].b, depth - 1, -beta, -alpha, 0);
    val = v;
    if (fabs(v) == inf)
        return -inf;
    if (beta <= v)
        return v;
    alpha = max(alpha, v);
    for (i = 1; i < canput; ++i){
        if (depth > simple_threshold)
            v = -nega_scout(lst[i].b, depth - 1, -alpha - window, -alpha, 0);
        else
            v = -nega_alpha(lst[i].b, depth - 1, -alpha - window, -alpha, 0);
        if (fabs(v) == inf)
            return -inf;
        if (beta <= v)
            return v;
        if (alpha < v){
            alpha = v;
            if (depth > simple_threshold)
                v = -nega_scout(lst[i].b, depth - 1, -beta, -alpha, 0);
            else
                v = -nega_alpha(lst[i].b, depth - 1, -beta, -alpha, 0);
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
        search_param.memo_ub[board] = val;
    else if (val >= beta)
        search_param.memo_lb[board] = val;
    else {
        search_param.memo_ub[board] = val;
        search_param.memo_lb[board] = val;
    }
    return val;
}

double map_double(double y1, double y2, double y3, double x){
    double a, b, c;
    double x1 = 4.0, x2 = 32.0, x3 = 64.0;
    a = ((y1 - y2) * (x1 - x3) - (y1 - y3) * (x1 - x2)) / ((x1 - x2) * (x1 - x3) * (x2 - x3));
    b = (y1 - y2) / (x1 - x2) - a * (x1 + x2);
    c = y1 - a * x1 * x1 - b * x1;
    return a * x * x + b * x + c;
}

double map_linar(double s, double e, double x){
    return s + (e - s) * x;
}

int cmp_main(board_priority_move p, board_priority_move q){
    return p.priority > q.priority;
}

int main(int argc, char* argv[]){
    int ansy, ansx, outy, outx, i, j, k, canput, former_depth = 7, former_vacant = hw2 - 4;
    double score, max_score;
    unsigned long long p, o;
    int board[board_index_num];
    int put;
    vector<board_priority_move> lst;
    int elem;
    int action_count;
    double game_ratio;
    int vacant_cnt, ai_player;
    int board_tmp;
    int y, x;
    
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
        canput = 0;
        for (i = 0; i < hw2; ++i){
            cin >> elem;
            vacant_cnt += (int)(elem == -1 || elem == 2);
            p |= (unsigned long long)(elem == ai_player) << i;
            o |= (unsigned long long)(elem == 1 - ai_player) << i;
        }
        for (i = 0; i < board_index_num; ++i){
            board_tmp = 0;
            for (j = 0; j < board_param.pattern_space[i]; ++j){
                if (1 & (p >> board_param.board_translate[i][j]))
                    board_tmp += board_param.pow3[j];
                else if (1 & (o >> board_param.board_translate[i][j]))
                    board_tmp += 2 * board_param.pow3[j];
            }
            board[i] = board_tmp;
        }
        search_param.min_max_depth = max(5, former_depth + vacant_cnt - former_vacant);
        //search_param.min_max_depth = 2;
        cerr << "start depth " << search_param.min_max_depth << endl;
        search_param.max_depth = search_param.min_max_depth;
        former_vacant = vacant_cnt;
        lst.clear();
        for (i = 0; i < board_index_num; ++i){
            for (j = 0; j < board_param.pattern_space[i]; ++j){
                if (board_param.legal[board[i]][j]){
                    ++canput;
                    board_priority_move tmp;
                    move(board, tmp.b, board_param.board_translate[i][j]);
                    tmp.priority = evaluate(tmp.b);
                    tmp.move = board_param.board_translate[i][j];
                    lst.push_back(tmp);
                }
            }
        }
        if (canput > 1)
            sort(lst.begin(), lst.end(), cmp_main);
        outy = -1;
        outx = -1;
        search_param.strt = tim();
        search_param.searched_nodes = 0;
        while (tim() - search_param.strt < search_param.tl / 2){
            search_param.memo_ub.clear();
            search_param.memo_lb.clear();
            search_param.turn = hw2 - vacant_cnt + search_param.max_depth;
            game_ratio = (double)search_param.turn / hw2;
            eval_param.pattern_weight = map_double(eval_param.weight_sme[0], eval_param.weight_sme[1], eval_param.weight_sme[2], game_ratio);
            eval_param.cnt_weight = map_double(eval_param.weight_sme[3], eval_param.weight_sme[4], eval_param.weight_sme[5], game_ratio);
            eval_param.canput_weight = map_double(eval_param.weight_sme[6], eval_param.weight_sme[7], eval_param.weight_sme[8], game_ratio);
            eval_param.cnt_bias = map_double(eval_param.weight_sme[9], eval_param.weight_sme[10], eval_param.weight_sme[11], game_ratio);
            max_score = -65000.0;
            for (i = 0; i < canput; ++i){
                score = -nega_scout(lst[i].b, search_param.max_depth - 1, -65000.0, -max_score, 0);
                if (fabs(score) == inf){
                    max_score = -inf;
                    break;
                }
                lst[i].priority = score;
                if (score > max_score){
                    max_score = score;
                    ansy = lst[i].move / hw;
                    ansx = lst[i].move % hw;
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
            cerr << "depth " << search_param.max_depth << " nodes " << search_param.searched_nodes << " nps " << ((unsigned long long)search_param.searched_nodes * 1000 / max(1, tim() - search_param.strt));
            for (i = 0; i < 1; ++i){
                cerr << "  " << (lst[i].move / hw) << (lst[i].move % hw) << " " << lst[i].priority;
            }
            cerr << " time " << tim() - search_param.strt << endl;
            if (vacant_cnt < search_param.max_depth || fabs(max_score) >= 1000.0){
                cerr << "game end" << endl;
                break;
            }
            ++search_param.max_depth;
        }
        cout << outy << " " << outx << endl;
    }
    return 0;
}