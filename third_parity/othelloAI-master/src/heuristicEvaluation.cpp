#include "heuristicEvaluation.h"

void ind2sub(const int sub,const int cols,const int rows,int *row,int *col) {
   *row=sub/cols;
   *col=sub%cols;
}

int stableDiscsHelper(vector <int> positions, unordered_set<int> & sDiscs, int corner, int symbol) {
    // only run on filled corners of symbol
    int nStableDiscs = 0;
    bool down, left;
    if (corner == 63) {
        down = false;
        left = true;
    } else if (corner == 56) {
        down = false;
        left = false;
    } else if (corner == 7) {
        down = true;
        left = true;
    } else {
        down = true;
        left = false;
    }
    int horInc, verInc;
    int horEnd, verEnd;
    if (down) {
        verInc = 8;
        verEnd = 56;
    } else {
        verInc = -8;
        verEnd = -56;
    }
    if (left) {
        horInc = -1;
        horEnd = -7;
    } else {
        horInc = 1;
        horEnd = 7;
    }

    for (int i = corner; i != corner + verEnd + verInc; i+=verInc) {
        if(positions[i] == symbol) {
            for (int j = i; j != i + horEnd; j+= horInc) {
                if (positions[j] == symbol) {
                    if (sDiscs.find(j) == sDiscs.end()) {
                        sDiscs.insert(j);
                    }
                } else {
                    break;
                }
            }
        } else if(positions[i] == -symbol) {
            if (positions[i+horEnd] == -symbol) {
                bool allIntStable = true;
                for (int j = i + horInc; j != i + horEnd; j+= horInc) {
                    if (positions[j] == symbol) {
                        ;
                    } else {
                        allIntStable = false;
                    }
                }
                if (allIntStable) {
                    for (int j = i + horInc; j != i + horEnd; j+= horInc) {
                        sDiscs.insert(j);
                    }
                }
            } else {
                break;
            } 
            if (positions[i+verEnd] == -symbol) {
                bool allIntStable = true;
                for (int j = i + verInc; j != i + verEnd; j+= verInc) {
                    if (positions[j] == symbol) {
                        ;
                    } else {
                        allIntStable = false;
                    }
                }
                if (allIntStable) {
                    for (int j = i + verInc; j != i + verEnd; j+= verInc) {
                        sDiscs.insert(j);
                    }
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

int stableDiscs(vector <int> positions,int symbol) {
    // grow out from for corners
    unordered_set<int> sDiscs;
    stableDiscsHelper(positions,sDiscs,63,symbol);
    stableDiscsHelper(positions,sDiscs,7,symbol);
    stableDiscsHelper(positions,sDiscs,0,symbol);
    stableDiscsHelper(positions,sDiscs,56,symbol);

    return sDiscs.size();
}


int potentialMobility(vector <int> positions, int symbol) {
    int interior[] = {9,10,11,12,13,15,17,18,19,20,21,22,25,26,27,28,29,30,33,34,35,36,37,38,41,42,43,44,45,46,49,50,51,52,53,54};
    int cur, top, bot, lef, rig, tr, tl, bl, br;
    int potMobilility;
    for (int i : interior) {
        cur = positions[i];
        lef = positions[i-1];
        rig = positions[i+1];
        top = positions[i-8];
        bot = positions[i+8];
        tr  = positions[i-7];
        tl  = positions[i-9];
        bl  = positions[i+7];
        br  = positions[i+9];
        // Potential Mobility 
        // # of discs that the oppenent has adjacent to empty squares
        potMobilility = 0;
        if (cur == -1 && (lef == 0 || rig == 0 || top == 0 || bot == 0 || tr == 0 || tl == 0 || bl == 0 || br == 0))
            potMobilility++;
            // # of empty squares adjacent ot the opponents discs
        if (cur = -symbol && lef == 0)
            potMobilility++;
        if (cur = -symbol && rig == 0)
            potMobilility++;
        if (cur = -symbol && top == 0)
            potMobilility++;
        if (cur = -symbol && bot == 0)
            potMobilility++;
        if (cur = -symbol && tr  == 0)
            potMobilility++;
        if (cur = -symbol && tl  == 0)
            potMobilility++;
        if (cur = -symbol && bl  == 0)
            potMobilility++;
        if (cur = -symbol && br  == 0)
            potMobilility++;
    }

        // go through exterior positions 
        // go through top positions
    int topa[] = {1,2,3,4,5,6};
    for (int i : topa) {
        cur = positions[i];
        lef = positions[i-1];
        rig = positions[i+1];
        if (cur == -symbol && (lef == 0 || rig == 0))
            potMobilility++;
            // # of empty squares adjacent ot the opponents discs
        if (cur = -symbol && lef == 0)
            potMobilility++;
        if (cur = -symbol && rig == 0)
            potMobilility++;
    }

        // go through bottom positions
    int bota[] = {57,58,59,60,61,62};
    for (int i : bota) {
        cur = positions[i];
        lef = positions[i-1];
        rig = positions[i+1];
        if (cur == -symbol && (lef == 0 || rig == 0))
            potMobilility++;
            // # of empty squares adjacent ot the opponents discs
        if (cur = -symbol && lef == 0)
            potMobilility++;
        if (cur = -symbol && rig == 0)
            potMobilility++;
    }

        // go through left positions
    int lefa[] = {8,16,24,32,40,48};
    for (int i : lefa) {
        cur = positions[i];
        bot = positions[i-8];
        top = positions[i+8];
        if (cur == -1 && (top == 0 || bot == 0 ))
            potMobilility++;
            // # of empty squares adjacent ot the opponents discs
        if (cur = -symbol && top == 0)
            potMobilility++;
        if (cur = -symbol && bot  == 0)
            potMobilility++;
    }

        // go through right positions
    int riga[] = {15,23,31,39,47,55};
    for (int i : riga) {
        cur = positions[i];
        bot = positions[i-8];
        top = positions[i+8];
        if (cur == -symbol && (top == 0 || bot == 0 ))
            potMobilility++;
            // # of empty squares adjacent ot the opponents discs
        if (cur = -symbol && top == 0)
            potMobilility++;
        if (cur = -symbol && bot  == 0)
            potMobilility++;
    }

    return potMobilility;
}

int heuristicEvaluation::heuristic5(othelloBoard board, int nSpacesRemaining,int symbol) {
    // do heuristic from perspective of player with p.symbol = 1
    int val = 0;
    vector <int> positions = board.positions;

    unordered_map<int, list<int>> validMoves;
    board.validMoves(validMoves,1);
    unordered_map<int, list<int>> otherValidMoves;
    board.validMoves(otherValidMoves,-1);

    //check if game is over     
    if ((validMoves.size() == 0) || (nSpacesRemaining == 0)) {   
        if ((otherValidMoves.size() == 0) || (nSpacesRemaining == 0)) {
            int s = accumulate(board.positions.begin(), board.positions.end(), 0);
            if (s > 0) {
                    // cout << "White wins!" << endl;
                return 100000*s;
            } else if (s < 0) {
                    // cout << "Black wins!" << endl;
                return 100000*s;
            } else {
                return 0;
            }
        }
    }

    // mobility
    int p = validMoves.size();
    int o = otherValidMoves.size();

    int mobility;
    if (p > o) {
        mobility = (100*p)/(p+o);
    } else if (p < o) {
        mobility = -(100*o)/(p+o);
    } else {
        mobility = 0;
    }

    int pPotMob = potentialMobility(positions, 1);
    int oPotMob = potentialMobility(positions, -1);

    int potMobility;
    if (pPotMob > oPotMob) {
        potMobility = (100*pPotMob)/(pPotMob+oPotMob);
    } else if (pPotMob < oPotMob) {
        potMobility = -(100*oPotMob)/(pPotMob+oPotMob);
    } else {
        potMobility = 0;
    }

    int corners;
    int cornerInds[4] = {0,7,56,63};

    int pCorners = 0;
    int oCorners = 0;

    for (auto i : cornerInds) {
        if (positions[i] == 1) {
            pCorners++;
        } else if (positions[i] == -1) {
            oCorners++;
        }
    } 

    if (pCorners > oCorners) {
        corners = 100 * (pCorners * pCorners) / (pCorners + oCorners);
    } else if(oCorners > pCorners) {
        corners = -100 * (oCorners * oCorners) / (pCorners + oCorners);
    } else {
        corners = 0;
    }

    vector <int> boardWeights = {
             100,-100, 100,  50,  50, 100,-100, 100,
            -100,-200, -50, -50, -50, -50,-200,-100,
             100, -50, 100,   0,   0, 100, -50, 100,
              50, -50,  0,    0,   0,   0, -50,  50,
              50, -50,  0,    0,   0,   0, -50,  50,
             100, -50, 100,   0,   0, 100, -50, 100,
            -100,-200, -50, -50, -50, -50,-200,-100, 
             100,-100, 100,  50,  50, 100,-100, 100
        };

    if (positions[0] != 0) {
        boardWeights[1]  = 0;
        boardWeights[2]  = 0;
        boardWeights[3]  = 0;

        boardWeights[8]  = 0;
        boardWeights[9]  = 0;
        boardWeights[10] = 0;
        boardWeights[11] = 0;

        boardWeights[16] = 0;
        boardWeights[17] = 0;
        boardWeights[18] = 0;

        boardWeights[24] = 0;
        boardWeights[25] = 0;
    }

    if (positions[7] != 0) {
        boardWeights[6]  = 0;
        boardWeights[5]  = 0;
        boardWeights[4]  = 0;

        boardWeights[15] = 0;
        boardWeights[14] = 0;
        boardWeights[13] = 0;
        boardWeights[12] = 0;

        boardWeights[23] = 0;
        boardWeights[22] = 0;
        boardWeights[21] = 0;

        boardWeights[31] = 0;
        boardWeights[30] = 0;
    }

    if (positions[56] != 0) {
        boardWeights[57] = 0;
        boardWeights[58] = 0;
        boardWeights[59] = 0;

        boardWeights[48] = 0;
        boardWeights[49] = 0;
        boardWeights[50] = 0;
        boardWeights[51] = 0;

        boardWeights[40] = 0;
        boardWeights[41] = 0;
        boardWeights[42] = 0;

        boardWeights[32] = 0;
        boardWeights[33] = 0;
    }

    if (positions[63] != 0) {
        boardWeights[62] = 0;
        boardWeights[61] = 0;
        boardWeights[60] = 0;

        boardWeights[55] = 0;
        boardWeights[54] = 0;
        boardWeights[53] = 0;
        boardWeights[52] = 0;

        boardWeights[47] = 0;
        boardWeights[46] = 0;
        boardWeights[45] = 0;

        boardWeights[39] = 0;
        boardWeights[38] = 0;
    }
 
    int naivety = inner_product(boardWeights.begin(), boardWeights.end(), positions.begin(), 0);

    int pStable = stableDiscs(positions,1);
    int oStable = stableDiscs(positions,-1);

    int stability;

    if (pStable > oStable) {
        stability = 100 * (pStable ) / (pStable + oStable);
    } else if(oStable > pStable) {
        stability = -100 * (oStable) / (pStable + oStable);
    } else {
        stability = 0;
    }

    // coin makeup
    int d = accumulate(board.positions.begin(), board.positions.end(), 0);
    int t = inner_product(board.positions.begin(),board.positions.end(),board.positions.begin(),0);
    int makeup = 100*d/t;

    // parity
    int parity;
    if (symbol == 1) {
        if (t % 2 == 0) {
            parity = -1;
        } else {
            parity = 1;
        }
    } else {
        if (t % 2 == 0) {
            parity = 1;
        } else {
            parity = -1;
        }
    }

    return 5*mobility + 2*potMobility + makeup + 5*naivety + 500*parity + 1000*corners + 1000*stability;
}


int heuristicEvaluation::heuristic(othelloBoard board, int nSpacesRemaining,int symbol) {
    return heuristic5(board, nSpacesRemaining,symbol);
}

heuristicEvaluation::heuristicEvaluation() {

}