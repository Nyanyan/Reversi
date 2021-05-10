#include <iostream>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <limits.h>
#include <assert.h>
#include <sstream>
#include "game.h"

float getTimeLimit() {
    float limit;
    bool validSelection = false;
    do {
        cout << "Pick time limit for computer: (seconds) ";        

        string str;
        cin >> str;
        istringstream iss(str);
        iss >> limit;

        if (iss.eof() == false) {
            cout << "Non-numeric input, please try again." << endl;
        } else if(limit < 0) {
            cout << "Negative numeric input, please try a positive number" << endl;
        } else {
            validSelection = true;
        }
    } while (!validSelection);

    return limit;
}

bool checkCPU(int id) {
    bool cpu;
    char c;
    bool validSelection = false;
    do {
        cout << "Is Player " << id << " a computer? y/n ";      

        string str;
        cin >> str;
        istringstream iss(str);
        iss >> c;

        cout << c << endl;

        iss.ignore(numeric_limits <streamsize> ::max(), '\n' );

        if (iss.eof() == false) {
            cout << "Non-single character input, please try again." << endl;
        } else 
        if(c != 'y' && c != 'n') {
            cout << "Did not enter 'y' or 'n'. Please try again." << endl;
        } else {
            validSelection = true;
        }
    } while (!validSelection);

    if (c == 'y') {
        cpu = true;
    } else {
        cpu = false;
    }

    return cpu;
}

int main (int argc, char *argv[]) {
    othelloBoard board;

    int choice;
    cout << "Load a game or start a new one?\n";
    cout << "1 -> Load a saved board state\n";
    cout << "2 -> Start a new game\n";

    bool validSelection = false;

    do {
        cout << "Selection: ";

        string str;
        cin >> str;
        istringstream iss(str);
        iss >> choice;

        if (iss.eof() == false) {
            cout << "Non-integer input, please try again." << endl;
        } else if(choice > 2 || choice < 0) {
            cout << "Integer selection out of range, please try again" << endl;
        } else {
            validSelection = true;
        }
    } while (!validSelection);

    othelloGame game (&board);  

    if (choice == 1)
        game.newGame = false;

    bool whiteMovesFirst = false;
    bool cpu1;
    bool cpu2;
    float limit;

    if (game.newGame) {

        cpu1 = checkCPU(1);
        cpu2 = checkCPU(2);

        if (cpu1 || cpu2) {
            limit = getTimeLimit();
        }   

        cout << "New Game\n";
        game.firstMove();

    } else {
        string filename;
        cout << "Give filename of savefile: ";
        cin >> filename;
        game.loadGame(filename, whiteMovesFirst, limit);

        cpu1 = checkCPU(1);
        cpu2 = checkCPU(2);
    }

    heuristicEvaluation h;

    // humanPlayer, playerId, n, symbol 
    player playerOne (!cpu1, 1, board.n,-1, h); // black
    player playerTwo (!cpu2, 0, board.n,1,  h);  // white

    if (cpu1 || cpu2) {
        playerOne.limit = limit;
        playerTwo.limit = limit;
    }

    if (whiteMovesFirst) {
        game.move(playerTwo);
        game.statusUpdate();
    }

    while (!game.complete) {
        game.move(playerOne); // player one moves
        game.move(playerTwo);
        game.statusUpdate(); // updates value of game.complete 
    };  
}
