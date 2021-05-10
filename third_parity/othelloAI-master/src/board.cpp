#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>
#include "board.h"

using namespace std;

othelloBoard::othelloBoard() {
    positions.assign(64,0);
}

// -1 is black, 1 is white
void othelloBoard::draw(unordered_map<int, list<int>> moves, int symbol) {
    cout << "  A B C D E F G H\n";
    int r = 0;
    for (int i = 0; i < n; i+=height ) {
        cout << ++r << "\e[48;5;34m\e[38;5;232m \033[0m";
        for (int j = i; j < i+width; j++) {
            if (positions[j] == 1) {
                cout << "\e[48;5;34m\e[38;5;256m\u25CF" << " \033[0m";
            } else if (positions[j] == -1) {
                cout << "\e[48;5;34m\e[38;5;232m\u25CF" << " \033[0m";
            } else if (moves.find(j) != moves.end() && symbol == -1) {
                cout << "\e[48;5;34m\e[38;5;232m\u2613" << " \033[0m";
            } else if (moves.find(j) != moves.end() && symbol == 1) {
                cout << "\e[48;5;34m\e[38;5;256m\u2613" << " \033[0m";
            } 
            else {
                cout << "\e[48;5;34m\e[38;5;232m\u00B7 \033[0m";
            }
        }
        cout << endl;
    }

    char alpha[9] = "ABCDEFGH";
    char nums[9]  = "12345678";
    int row, col;
    int i = 1;

    for (auto kv : moves) {
        ind2sub(kv.first, width, height, &row, &col);
        cout << "Possible Move " << i << ": " << alpha[col] << nums[row] << ", ";
        list<int> l = kv.second;
        cout << "Pieces to be flipped:";
        for (auto k : l) {
            ind2sub(k, width, height, &row, &col);
            cout << " "<< alpha[col] << nums[row] << ",";
        }

        cout << endl;
        i++;
    }
}

void othelloBoard::ind2sub(const int sub,const int cols,const int rows,int *row,int *col) {
   *row=sub/cols;
   *col=sub%cols;
}

void othelloBoard::validMovesHelper(const int & clr, const int & i, const int & inc, unordered_map<int, list<int>> & pieces, list <int> & oldcandidates, pair<int,list<int>> & move) {
    list<int> candidates;
    int crow, ccol, prow, pcol;
    int pos;

    for (int j = inc; (i + j < n) && (i + j > -1); j+=inc) {
        // first check to make sure don't wrap around board
        // check to see that diff in cols and rows doesn't exceed 1
        ind2sub(i+j-inc, width, height, &prow, &pcol);
        ind2sub(i+j,     width, height, &crow, &ccol);
        if (abs(crow - prow) > 1 || abs(ccol - pcol) > 1)
            break;

        pos = positions[i+j];
        if (pos == clr)
            break;
        if (pos == -clr) {
            candidates.push_front(i+j);
            continue;
        }
        if (pos == 0 && positions[i+j-inc] == clr)
            break;
        if (pos == 0 && positions[i+j-inc] == -clr) { 
            if (pieces.find(i+j) != pieces.end()) {
                oldcandidates = pieces[i+j];
                oldcandidates.splice(oldcandidates.begin(),candidates);
                pieces.erase(i+j);
                move.first = i+j;
                move.second = oldcandidates;
                pieces.insert(move);
            } else {
                move.first = i+j;
                move.second = candidates;
                pieces.insert(move);
            }
            break;
        }
    }
}

void othelloBoard::validMoves (unordered_map<int, list<int>> & moves, int symbol) {
    int incs[8] = {8,-8,1,-1,-7,7,-9,9};
    list <int> oldcandidates;
    pair<int,list<int>> move;
    for (int i = 0; i < n; i+=1) {
            if (positions[i] == 0)
                continue;
            if (positions[i] == symbol) {
                // go through columns
                validMovesHelper(symbol, i, incs[0], moves, oldcandidates, move);
                validMovesHelper(symbol, i, incs[1], moves, oldcandidates, move);
                // go through rows
                validMovesHelper(symbol, i, incs[2], moves, oldcandidates, move);
                validMovesHelper(symbol, i, incs[3], moves, oldcandidates, move);
                // go through diagnols
                validMovesHelper(symbol, i, incs[4], moves, oldcandidates, move);
                validMovesHelper(symbol, i, incs[5], moves, oldcandidates, move);
                validMovesHelper(symbol, i, incs[6], moves, oldcandidates, move);
                validMovesHelper(symbol, i, incs[7], moves, oldcandidates, move);
            }
            if (positions[i] == -symbol)
                continue;
        }
}

void othelloBoard::updatePositions (pair<int, list<int>> move, int symbol) {

    int piece = move.first;
    pastMoves.push_back(piece);

    positions[piece] = symbol;

    list<int> l = move.second;

    for (auto k : l) {
        positions[k] = symbol;
    }

    nMoves++;
}