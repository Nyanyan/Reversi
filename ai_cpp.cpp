#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx")

// Reversi AI C++ version 3
// previous 12th rate 29.19

#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <unordered_map>
#include <random>

using namespace std;

#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw2_m1 63
#define hw2_mhw 56
#define window 0.00001
#define simple_threshold 4
#define inf 100000.0
#define param_num 36
#define board_index_num 38
#define pattern_num 5

struct hash_arr{
    static size_t m_hash_arr_random;
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
        seed ^= m_hash_arr_random + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

size_t hash_arr::m_hash_arr_random = (size_t) random_device()();

struct board_param{
    unsigned long long trans[board_index_num][6561][hw];
    unsigned long long neighbor8[board_index_num][6561][hw];
    bool legal[6561][hw];
    int put[hw2][board_index_num];
    int board_translate[board_index_num][8];
    int board_rev_translate[hw2][4][2];
    int pattern_space[board_index_num];
    int reverse[6561];
    int pow3[15];
    int rev_bit3[6561][8];
    int pop_digit[6561][8];
    int digit_pow[3][10];
};

struct eval_param{
    double weight[hw2];
    double pattern_weight, cnt_weight, canput_weight, weight_weight, confirm_weight, pot_canput_weight, open_weight;
    double cnt_bias;
    double weight_sme[param_num];
    double avg_canput[hw2];
    int canput[6561];
    int cnt_p[6561], cnt_o[6561];
    double weight_p[hw][6561], weight_o[hw][6561];
    int pattern_variation[pattern_num], pattern_space[pattern_num];
    int pattern_translate[pattern_num][8][10][2];
    double pattern_each_weight[pattern_num];
    double pattern[pattern_num][59049];
    int confirm_p[6561], confirm_o[6561];
    int pot_canput_p[6561], pot_canput_o[6561];
    double open_eval[40];
};

struct search_param{
    int max_depth;
    int min_max_depth;
    int strt, tl;
    int turn;
    int searched_nodes;
    int vacant_lst[hw2];
    int vacant_cnt;
};

struct board_priority_move{
    int b[board_index_num];
    double priority;
    int move;
    double open_val;
};

struct board_priority{
    int b[board_index_num];
    double priority;
    double n_open_val;
};

board_param board_param;
eval_param eval_param;
search_param search_param;

int xorx=123456789, xory=362436069, xorz=521288629, xorw=88675123;
inline double myrandom(){
    int t = (xorx^(xorx<<11));
    xorx = xory;
    xory = xorz;
    xorz = xorw;
    xorw = xorw=(xorw^(xorw>>19))^(t^(t>>8));
    return (double)(xorw) / 2147483648.0;
}

inline int randint(int fr, int to){
    return fr + (int)(myrandom() * (to - fr + 1));
}

#define prob_num_step 10000
#define prob_step_width (2.0 / prob_num_step)
double prob_arr[prob_num_step];

void prob_init(){
    double x;
    for (int idx = 0; idx < prob_num_step; idx++){
        x = -prob_step_width * idx;
        prob_arr[idx] = exp(x * 1.5);
    }
}
double prob(double dis){
    if (dis >= 0)
        return 1.0;
    //return exp(dis * 1.5);
    return prob_arr[min(prob_num_step - 1, (int)(-dis / prob_step_width))];
}

inline int tim(){
    return static_cast<int>(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count());
}

void print_board_line(int tmp){
    int j;
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

inline int check_mobility(const int p, const int o){
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
    char cbuf[1024];
    int i, j, k, l;
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
    if (argc > 1)
        file = argv[1];
    else
        file = "param.txt";
    if ((fp = fopen(file, "r")) == NULL){
        printf("param file not exist");
        exit(1);
    }
    for (i = 0; i < hw2; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        eval_param.avg_canput[i] = atof(cbuf);
    }
    for (i = 0; i < 10; i++){
        if (!fgets(cbuf, 1024, fp)){
            printf("param file broken");
            exit(1);
        }
        weight_buf[i] = atof(cbuf);
    }
    for (i = 0; i < hw2; i++)
        eval_param.weight[i] = weight_buf[translate[i]];
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
    for (i = 0; i < pattern_num; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("const.txt broken");
            exit(1);
        }
        eval_param.pattern_space[i] = atoi(cbuf);
    }
    for (i = 0; i < pattern_num; ++i){
        if (!fgets(cbuf, 1024, fp)){
            printf("const.txt broken");
            exit(1);
        }
        eval_param.pattern_variation[i] = atoi(cbuf);
    }
    for (i = 0; i < pattern_num; ++i){
        for (j = 0; j < eval_param.pattern_variation[i]; ++j){
            for (k = 0; k < eval_param.pattern_space[i]; ++k){
                if (!fgets(cbuf, 1024, fp)){
                    printf("const.txt broken");
                    exit(1);
                }
                eval_param.pattern_translate[i][j][k][0] = atoi(cbuf) / hw;
                eval_param.pattern_translate[i][j][k][1] = atoi(cbuf) % hw;
            }
        }
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
            eval_param.pattern[i][j] = atof(cbuf);
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
                    board_param.neighbor8[k][i][j] = 0;
                    board_param.neighbor8[k][i][j] |= (0b0111111001111110011111100111111001111110011111100111111001111110 & board_param.trans[k][i][j]) << 1;
                    board_param.neighbor8[k][i][j] |= (0b0111111001111110011111100111111001111110011111100111111001111110 & board_param.trans[k][i][j]) >> 1;
                    board_param.neighbor8[k][i][j] |= (0b0000000011111111111111111111111111111111111111111111111100000000 & board_param.trans[k][i][j]) << hw;
                    board_param.neighbor8[k][i][j] |= (0b0000000011111111111111111111111111111111111111111111111100000000 & board_param.trans[k][i][j]) >> hw;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) << hw_m1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) >> hw_m1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) << hw_p1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) >> hw_p1;
                    board_param.neighbor8[k][i][j] &= ~board_param.trans[k][i][j];
                }
            } else
                board_param.legal[i][j] = false;
        }
        eval_param.canput[i] = canput_num;
    }
    for (i = 0; i < 15; ++i)
        board_param.pow3[i] = (int)pow(3, i);
    for (i = 0; i < 6561; ++i){
        for (j = 0; j < 8; ++j){
            board_param.rev_bit3[i][j] = board_param.pow3[j] * (2 - (i / board_param.pow3[j]) % 3);
            board_param.pop_digit[i][j] = i / board_param.pow3[j] % 3;
        }
    }
    for (i = 0; i < hw; ++i){
        for (j = 0; j < 6561; ++j){
            eval_param.weight_p[i][j] = 0.0;
            eval_param.weight_o[i][j] = 0.0;
            for (k = 0; k < 8; ++k){
                if (board_param.pop_digit[j][k] == 1)
                    eval_param.weight_p[i][j] += eval_param.weight[i * hw + k];
                else if (board_param.pop_digit[j][k] == 2)
                    eval_param.weight_o[i][j] += eval_param.weight[i * hw + k];
            }
        }
    }
    bool flag;
    for (i = 0; i < 6561; ++i){
        eval_param.confirm_p[i] = 0;
        eval_param.confirm_o[i] = 0;
        flag = true;
        for (j = 0; j < hw; ++j)
            if (!board_param.pop_digit[i][j])
                flag = false;
        if (flag){
            for (j = 0; j < hw; ++j){
                if (board_param.pop_digit[i][j] == 1)
                    ++eval_param.confirm_p[i];
                else
                    ++eval_param.confirm_o[i];
            }
        } else {
            flag = true;
            for (j = 0; j < hw; ++j){
                if (board_param.pop_digit[i][j] != 1)
                    break;
                ++eval_param.confirm_p[i];
                if (k == hw_m1)
                    flag = false;
            }
            if (flag){
                for (j = hw_m1; j >= 0; --j){
                    if (board_param.pop_digit[i][j] != 1)
                        break;
                    ++eval_param.confirm_p[i];
                    if (k == hw_m1)
                        flag = false;
                }
            }
            flag = true;
            for (j = 0; j < hw; ++j){
                if (board_param.pop_digit[i][j] != 2)
                    break;
                ++eval_param.confirm_o[i];
                if (k == hw_m1)
                    flag = false;
            }
            if (flag){
                for (j = hw_m1; j >= 0; --j){
                    if (board_param.pop_digit[i][j] != 2)
                        break;
                    ++eval_param.confirm_o[i];
                    if (k == hw_m1)
                        flag = false;
                }
            }
        }
    }
    for (i = 0; i < 6561; ++i){
        eval_param.pot_canput_p[i] = 0;
        eval_param.pot_canput_o[i] = 0;
        for (j = 0; j < hw_m1; ++j){
            if (board_param.pop_digit[i][j] == 0){
                if (board_param.pop_digit[i][j + 1] == 2)
                    ++eval_param.pot_canput_p[i];
                else if (board_param.pop_digit[i][j + 1] == 1)
                    ++eval_param.pot_canput_o[i];
            }
        }
        for (j = 1; j < hw; ++j){
            if (board_param.pop_digit[i][j] == 0){
                if (board_param.pop_digit[i][j - 1] == 2)
                    ++eval_param.pot_canput_p[i];
                else if (board_param.pop_digit[i][j - 1] == 1)
                    ++eval_param.pot_canput_o[i];
            }
        }
    }
    for (i = 0; i < 3; ++i){
        for (j = 0; j < 10; ++j)
            board_param.digit_pow[i][j] = i * board_param.pow3[j];
    }
    for (i = 0; i < 40; ++i)
        eval_param.open_eval[i] = min(1.0, pow(2.0, 2.0 - 0.667 * i) - 1.0);
}

inline double pattern_eval(const int *board){
    int i, j, k, tmp;
    double res = 0.0;
    for (i = 0; i < 3; ++i){
        for (j = 0; j < eval_param.pattern_variation[i]; ++j){
            tmp = 0;
            for (k = 0; k < eval_param.pattern_space[i]; ++k)
                tmp += board_param.digit_pow[board_param.pop_digit[board[eval_param.pattern_translate[i][j][k][0]]][eval_param.pattern_translate[i][j][k][1]]][k];
            res += eval_param.pattern[i][tmp] * eval_param.pattern_each_weight[i];
        }
    }
    res += eval_param.pattern[3][board[21]] * eval_param.pattern_each_weight[3];
    res += eval_param.pattern[3][board[32]] * eval_param.pattern_each_weight[3];
    res += eval_param.pattern[4][board[0]] * eval_param.pattern_each_weight[4];
    res += eval_param.pattern[4][board[7]] * eval_param.pattern_each_weight[4];
    res += eval_param.pattern[4][board[8]] * eval_param.pattern_each_weight[4];
    res += eval_param.pattern[4][board[15]] * eval_param.pattern_each_weight[4];
    return res;
}

inline double canput_eval(const int *board){
    int i;
    int res = 0;
    for (i = 0; i < board_index_num; ++i)
        res += eval_param.canput[board[i]];
    return ((double)res - eval_param.avg_canput[search_param.turn]) / max(1.0, (double)res + eval_param.avg_canput[search_param.turn]);
}

inline double cnt_eval(const int *board){
    int i;
    int cnt_p = 0, cnt_o = 0;
    for (i = 0; i < hw; ++i){
        cnt_p += eval_param.cnt_p[board[i]];
        cnt_o += eval_param.cnt_o[board[i]];
    }
    return ((double)cnt_p - cnt_o) / max(1, cnt_p + cnt_o);
}

inline double weight_eval(const int *board){
    int i;
    double res_p = 0.0, res_o = 0.0;
    int cnt_p = 0, cnt_o = 0;
    for (i = 0; i < hw; ++i){
        res_p += eval_param.weight_p[i][board[i]];
        res_o += eval_param.weight_o[i][board[i]];
    }
    return (res_p - res_o) / (abs(res_p) + abs(res_o));
}

inline double confirm_eval(const int *board){
    int res_p, res_o;
    res_p = eval_param.confirm_p[board[0]];
    res_p += eval_param.confirm_p[board[7]];
    res_p += eval_param.confirm_p[board[8]];
    res_p += eval_param.confirm_p[board[15]];
    res_o = eval_param.confirm_o[board[0]];
    res_o += eval_param.confirm_o[board[7]];
    res_o += eval_param.confirm_o[board[8]];
    res_o += eval_param.confirm_o[board[15]];
    return (double)(res_p - res_o) / max(1, res_p + res_o);
}

inline double pot_canput_eval(const int *board){
    int i;
    int res_p = 0, res_o = 0;
    for (i = 0; i < board_index_num; ++i){
        res_p += eval_param.pot_canput_p[board[i]];
        res_o += eval_param.pot_canput_o[board[i]];
    }
    return (double)(res_p - res_o) / max(1, res_p + res_o);
}

inline double evaluate(const int *board, const double open){
    double pattern = pattern_eval(board);
    double cnt = cnt_eval(board);
    double canput = canput_eval(board);
    double weight = weight_eval(board);
    double confirm = confirm_eval(board);
    double pot_canput = pot_canput_eval(board);
    //double open = eval_param.open_eval[min(39, open_val)];
    return 
        pattern * eval_param.pattern_weight + 
        cnt * eval_param.cnt_weight + 
        canput * eval_param.canput_weight + 
        weight * eval_param.weight_weight + 
        confirm * eval_param.confirm_weight + 
        pot_canput * eval_param.pot_canput_weight + 
        open * eval_param.open_weight;
}

inline double end_game(const int *board){
    int res = 0, i, j, p, o;
    for (i = 0; i < hw; ++i){
        res += eval_param.cnt_p[board[i]];
        res -= eval_param.cnt_o[board[i]];
    }
    return (double)res * 1000.0;
}

inline int move_open(int *board, int (&res)[board_index_num], int coord){
    int i, j, tmp;
    unsigned long long rev = 0, neighbor = 0;
    for (i = 0; i < board_index_num; ++i){
        res[i] = board_param.reverse[board[i]];
        if (board_param.put[coord][i] != -1){
            rev |= board_param.trans[i][board[i]][board_param.put[coord][i]];
            neighbor |= board_param.neighbor8[i][board[i]][board_param.put[coord][i]];
        }
    }
    for (i = 0; i < hw2; ++i){
        if (1 & (rev >> i)){
            for (j = 0; j < 4; ++j){
                if (board_param.board_rev_translate[i][j][0] == -1)
                    break;
                res[board_param.board_rev_translate[i][j][0]] += board_param.rev_bit3[res[board_param.board_rev_translate[i][j][0]]][board_param.board_rev_translate[i][j][1]];
            }
        }
    }
    int open_val = 0;
    for (i = 0; i < hw2; ++i){
        if(1 & (neighbor >> i))
            open_val += (int)(board_param.pop_digit[board[i >> 3]][i & 0b111] == 0);
    }
    return open_val;
}

int cmp(board_priority p, board_priority q){
    return p.priority > q.priority;
}

double nega_alpha(int *board, const int depth, double alpha, double beta, const int skip_cnt, double p_open_val, double o_open_val, int p_cnt, int o_cnt){
    ++search_param.searched_nodes;
    if (tim() - search_param.strt > search_param.tl)
        return -inf;
    if (skip_cnt == 2)
        return end_game(board);
    else if (depth == 0)
        return evaluate(board, p_open_val / max(1, p_cnt) - o_open_val / max(1, o_cnt));
    bool is_pass = true;
    int i, j, k;
    double val = -65000.0, v;
    int n_board[board_index_num];
    ++p_cnt;
    for (j = 0; j < search_param.vacant_cnt; ++j){
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[search_param.vacant_lst[j]][i] != -1){
                if (board_param.legal[board[i]][board_param.put[search_param.vacant_lst[j]][i]]){
                    is_pass = false;
                    v = -nega_alpha(n_board, depth - 1, -beta, -alpha, 0, o_open_val, p_open_val + eval_param.open_eval[move_open(board, n_board, search_param.vacant_lst[j])], o_cnt, p_cnt);
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
        return -nega_alpha(n_board, depth, -beta, -alpha, skip_cnt + 1, o_open_val, p_open_val, o_cnt, p_cnt - 1);
    }
    return val;
}

double nega_alpha_null_window(int *board, const int depth, double alpha, double beta, const int skip_cnt, double p_open_val, double o_open_val, int p_cnt, int o_cnt){
    ++search_param.searched_nodes;
    if (tim() - search_param.strt > search_param.tl)
        return -inf;
    if (skip_cnt == 2)
        return end_game(board);
    else if (depth == 0)
        return evaluate(board, p_open_val / max(1, p_cnt) - o_open_val / max(1, o_cnt));
    int i, j, k, canput = 0;
    double val = -65000.0, v;
    board_priority lst[30];
    ++p_cnt;
    for (j = 0; j < search_param.vacant_cnt; ++j){
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[search_param.vacant_lst[j]][i] != -1){
                if (board_param.legal[board[i]][board_param.put[search_param.vacant_lst[j]][i]]){
                    lst[canput].n_open_val = p_open_val + eval_param.open_eval[move_open(board, lst[canput].b, search_param.vacant_lst[j])];
                    lst[canput].priority = lst[canput].n_open_val / p_cnt - o_open_val / max(1, o_cnt);
                    ++canput;
                    break;
                }
            }
        }
    }
    if (canput == 0){
        int n_board[board_index_num];
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_alpha_null_window(n_board, depth, -beta, -alpha, skip_cnt + 1, o_open_val, p_open_val, o_cnt, p_cnt - 1);
    }
    if (canput > 1)
        sort(lst, lst + canput, cmp);
    for (i = 0; i < canput; ++i){
        if (search_param.turn < 45 && depth < search_param.max_depth - 2 && prob(lst[i].priority) < myrandom() && prob(lst[i].n_open_val / p_cnt - lst[0].n_open_val / p_cnt) < myrandom())
            continue;
        if (depth > simple_threshold)
            v = -nega_alpha_null_window(lst[i].b, depth - 1, -beta, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, p_cnt);
        else
            v = -nega_alpha(lst[i].b, depth - 1, -beta, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, p_cnt);
        if (fabs(v) == inf)
            return -inf;
        if (beta <= v)
            return v;
        alpha = max(alpha, v);
        if (val < v)
            val = v;
    }
    return val;
}

double nega_scout(int *board, const int depth, double alpha, double beta, const int skip_cnt, double p_open_val, double o_open_val, int p_cnt, int o_cnt){
    ++search_param.searched_nodes;
    if (tim() - search_param.strt > search_param.tl)
        return -inf;
    if (skip_cnt == 2)
        return end_game(board);
    int i, j, k, canput = 0;
    double val = -65000.0, v;
    board_priority lst[30];
    ++p_cnt;
    for (j = 0; j < search_param.vacant_cnt; ++j){
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[search_param.vacant_lst[j]][i] != -1){
                if (board_param.legal[board[i]][board_param.put[search_param.vacant_lst[j]][i]]){
                    lst[canput].n_open_val = p_open_val + eval_param.open_eval[move_open(board, lst[canput].b, search_param.vacant_lst[j])];
                    lst[canput].priority = lst[canput].n_open_val / p_cnt - o_open_val / max(1, o_cnt);
                    ++canput;
                    break;
                }
            }
        }
    }
    if (canput == 0){
        int n_board[board_index_num];
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_scout(n_board, depth, -beta, -alpha, skip_cnt + 1, o_open_val, p_open_val, o_cnt, p_cnt - 1);
    }
    if (canput > 1)
        sort(lst, lst + canput, cmp);
    if (depth > simple_threshold)
        v = -nega_scout(lst[0].b, depth - 1, -beta, -alpha, 0, o_open_val, lst[0].n_open_val, o_cnt, p_cnt);
    else
        v = -nega_alpha(lst[0].b, depth - 1, -beta, -alpha, 0, o_open_val, lst[0].n_open_val, o_cnt, p_cnt);
    val = v;
    if (fabs(v) == inf)
        return -inf;
    if (beta <= v)
        return v;
    alpha = max(alpha, v);
    for (i = 1; i < canput; ++i){
        if (search_param.turn < 45 && depth < search_param.max_depth - 2 && prob(lst[i].priority) < myrandom() && prob(lst[i].n_open_val / p_cnt - lst[0].n_open_val / p_cnt) < myrandom())
            continue;
        v = -nega_alpha_null_window(lst[i].b, depth - 1, -alpha - window, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, p_cnt);
        if (fabs(v) == inf)
            return -inf;
        if (beta <= v)
            return v;
        if (alpha < v){
            alpha = v;
            if (depth > simple_threshold)
                v = -nega_scout(lst[i].b, depth - 1, -beta, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, p_cnt);
            else
                v = -nega_alpha(lst[i].b, depth - 1, -beta, -alpha, 0, o_open_val, lst[i].n_open_val, o_cnt, p_cnt);
            if (fabs(v) == inf)
                return -inf;
            if (beta <= v)
                return v;
            alpha = max(alpha, v);
        }
        if (val < v)
            val = v;
    }
    return val;
}

double map_double(double y1, double y2, double y3, double x){
    double a, b, c;
    double x1 = 4.0 / hw2, x2 = 25.0 / hw2, x3 = 64.0 / hw2;
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
    int ai_player;
    int board_tmp;
    int y, x;
    double final_score;
    int board_size;
    bool first_turn = true;
    string action;

    init(argc, argv);
    prob_init();
    init(argc, argv);
    cin >> ai_player;
    cin >> search_param.tl;
    
    if (ai_player == 0){
        cerr << "AI initialized AI is Black timeout in " << search_param.tl << " ms" << endl;
    }else{
        cerr << "AI initialized AI is White timeout in " << search_param.tl << " ms" << endl;
    }
    while (true){
        outy = -1;
        outx = -1;
        search_param.vacant_cnt = 0;
        p = 0;
        o = 0;
        canput = 0;
        for (i = 0; i < hw2; ++i){
            cin >> elem;
            if (elem == -1 || elem == 2)
                search_param.vacant_lst[search_param.vacant_cnt++] = i;
            p |= (unsigned long long)(elem == ai_player) << i;
            o |= (unsigned long long)(elem == 1 - ai_player) << i;
        }
        search_param.strt = tim();
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
        search_param.min_max_depth = max(5, former_depth + search_param.vacant_cnt - former_vacant);            
        search_param.max_depth = search_param.min_max_depth;
        former_vacant = search_param.vacant_cnt;
        lst.clear();
        for (j = 0; j < search_param.vacant_cnt; ++j){
            for (i = 0; i < board_index_num; ++i){
                if (board_param.put[search_param.vacant_lst[j]][i] != -1){
                    if (board_param.legal[board[i]][board_param.put[search_param.vacant_lst[j]][i]]){
                        ++canput;
                        board_priority_move tmp;
                        tmp.open_val = eval_param.open_eval[move_open(board, tmp.b, search_param.vacant_lst[j])];
                        tmp.priority = tmp.open_val;
                        tmp.move = search_param.vacant_lst[j];
                        lst.push_back(tmp);
                        break;
                    }
                }
            }
        }
        cerr << "canput " << canput << endl;
        for (i = 0; i < canput; ++i)
            cerr << lst[i].move << " ";
        cerr << endl;
        if (canput > 1)
            sort(lst.begin(), lst.end(), cmp_main);
        outy = -1;
        outx = -1;
        search_param.searched_nodes = 0;
        while (tim() - search_param.strt < search_param.tl){
            search_param.turn = min(63, hw2 - search_param.vacant_cnt + search_param.max_depth);
            game_ratio = (double)search_param.turn / hw2;
            eval_param.pattern_weight = map_double(eval_param.weight_sme[0], eval_param.weight_sme[1], eval_param.weight_sme[2], game_ratio);
            eval_param.cnt_weight = map_double(eval_param.weight_sme[3], eval_param.weight_sme[4], eval_param.weight_sme[5], game_ratio);
            eval_param.canput_weight = map_double(eval_param.weight_sme[6], eval_param.weight_sme[7], eval_param.weight_sme[8], game_ratio);
            eval_param.weight_weight = map_double(eval_param.weight_sme[9], eval_param.weight_sme[10], eval_param.weight_sme[11], game_ratio);
            eval_param.confirm_weight = map_double(eval_param.weight_sme[12], eval_param.weight_sme[13], eval_param.weight_sme[14], game_ratio);
            eval_param.pot_canput_weight = map_double(eval_param.weight_sme[15], eval_param.weight_sme[16], eval_param.weight_sme[17], game_ratio);
            eval_param.open_weight = map_double(eval_param.weight_sme[18], eval_param.weight_sme[19], eval_param.weight_sme[20], game_ratio);
            for (i = 0; i < pattern_num; ++i)
                eval_param.pattern_each_weight[i] = map_double(eval_param.weight_sme[21 + i * 3], eval_param.weight_sme[22 + i * 3], eval_param.weight_sme[23 + i * 3], game_ratio);
            score = -nega_scout(lst[0].b, search_param.max_depth - 1, -65000.0, 65000.0, 0, 0.0, 0.0, 0, 0);
            if (fabs(score) == inf){
                max_score = -inf;
            } else {
                lst[0].priority = score;
                max_score = score;
                ansy = lst[0].move / hw;
                ansx = lst[0].move % hw;
                for (i = 1; i < canput; ++i){
                    score = -nega_alpha_null_window(lst[i].b, search_param.max_depth - 1, -max_score - window, -max_score, 0, 0.0, lst[i].open_val, 0, 1);
                    if (score > max_score){
                        score = -nega_scout(lst[i].b, search_param.max_depth - 1, -65000.0, -score, 0, 0.0, lst[i].open_val, 0, 1);
                        if (fabs(score) == inf){
                            max_score = -inf;
                            break;
                        }
                        lst[i].priority = score;
                        max_score = score;
                        ansy = lst[i].move / hw;
                        ansx = lst[i].move % hw;
                    } else
                        lst[i].priority = score;
                }
            }
            if (max_score == -inf){
                cerr << "depth " << search_param.max_depth << " timeoout" << endl;
                break;
            }
            final_score = max_score;
            former_depth = search_param.max_depth;
            outy = ansy;
            outx = ansx;
            if (canput > 1)
                sort(lst.begin(), lst.end(), cmp_main);
            cerr << "depth " << search_param.max_depth << " nodes " << search_param.searched_nodes << " nps " << ((unsigned long long)search_param.searched_nodes * 1000 / max(1, tim() - search_param.strt));
            cerr << "  " << (lst[0].move / hw) << (lst[0].move % hw) << " " << lst[0].priority;
            cerr << " time " << tim() - search_param.strt << endl;
            if (fabs(max_score) >= 1000.0 || search_param.max_depth >= hw2){
                cerr << "game end" << endl;
                break;
            }
            ++search_param.max_depth;
        }
        cerr << (char)(outx + 97) << (outy + 1) << endl;
        cerr << tim() - search_param.strt << endl;
        cout << outy << " " << outx << endl;
    }
    return 0;
}
