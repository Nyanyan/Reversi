#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include "game.h"

using namespace std;

void ind2subg(const int sub,const int cols,const int rows,int *row,int *col) {
   *row=sub/cols;
   *col=sub%cols;
}


othelloGame::othelloGame(othelloBoard* a) {
    board = a; 
}

void othelloGame::firstMove() {
    vector<int> pos(board->n,0);

    pos[27] = 1;
    pos[28] = -1;
    pos[35] = -1;
    pos[36] = 1;
    board->positions.swap(pos);
}

void othelloGame::loadGame(string gameFileName, bool & whiteMovesFirst, float & limit) {
    vector<int> pos(board->n,0);
    ifstream gameFile;
    string line;

    try {
        gameFile.open(gameFileName);
    }
    catch (ios_base::failure& e) {
        cerr << e.what() << '\n';
    }

    int ind = 0;
    for (int j = 0; j < 8; j++) {
        getline(gameFile,line);
        for (int i = 0; i < 16; i+=2) {
            char c = line[i];
            if (c == 'B')
                pos[ind] = -1;
            if (c == 'W')
                pos[ind] = 1;
            if (c == '0')
                pos[ind] = 0;
            ind++;
        }
    }

    getline(gameFile,line);
    char c = line[0];
    if (c == 'W') {
        whiteMovesFirst = true;
    }

    getline(gameFile,line);

    limit = stof(line);

    board->nMoves = inner_product(pos.begin(),pos.end(),pos.begin(),0);
    cout << "Board n moves " << board->nMoves << endl;
    board->positions.swap(pos);
    gameFile.close();
}


void othelloGame::move(player p) {
    unordered_map<int, list<int>> moves;
    board->validMoves(moves,p.symbol);
    if (moves.empty()) {
        passes[p.playerId] = 1;
    } else {
        passes[p.playerId] = 0;
        board->draw(moves,p.symbol);
        pair<int, list<int>> move = p.selectMove(*board,moves);

        char alpha[9] = "ABCDEFGH";
        char nums[9]  = "12345678";
        int row, col;
        ind2subg(move.first, 8, 8, &row, &col);
        cout << "Move selected is at: " << alpha[col] << nums[row] << ',' << move.first << endl << endl;

        board->updatePositions(move,p.symbol);
    }
}

void othelloGame::statusUpdate() {
    if (passes[0] + passes[1] == 2) {
        complete = true;
        unordered_map<int, list<int>> moves;
        board->draw(moves,1);
        int s  = accumulate(board->positions.begin(), board->positions.end(), 0);
        cout << "Game is complete.\n";
        cout << "Score: " << s << endl;
        if (s < 0) {
            cout << "Black wins!" << endl;
        } else if (s > 0) {
            cout << "White wins!" << endl;
        } else {
            cout << "Tie!" << endl;
        }
    } else {
        complete = false;
    }
}