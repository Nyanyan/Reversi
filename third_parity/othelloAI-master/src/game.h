#include "player.h"

using namespace std;

class othelloGame {

othelloBoard* board;

public:
    bool complete = false; // boolean if game is complete or not.
    bool newGame = true;
    int passes[2];

    othelloGame(othelloBoard* a);

	void loadGame(string gameFileName, bool & whiteMovesFirst, float & limit);
    void firstMove();

    void move(player p);

    void statusUpdate();
};