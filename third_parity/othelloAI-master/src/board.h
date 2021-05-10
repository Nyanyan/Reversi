#include <list>

using namespace std;

class othelloBoard {
private:
    const static int width  = 8;
    const static int height = 8;

    void validMovesHelper(const int & clr, const int & i, const int & inc, unordered_map<int, list<int>> & pieces, list <int> & oldcandidates, pair<int,list<int>> & move);
    void ind2sub(const int sub,const int cols,const int rows,int *row,int *col);

public: 
    const static int n = width*height;
    vector<int> positions;
    list<int> pastMoves;

    int nMoves = 0;

    othelloBoard();

    void draw(unordered_map<int, list<int>> moves, int symbol);
    void validMoves (unordered_map<int, list<int>> & moves, int symbol);

    void updatePositions(pair<int, list<int>> move, int symbol);
};