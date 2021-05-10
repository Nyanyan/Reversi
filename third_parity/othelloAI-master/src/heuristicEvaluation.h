#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <limits>
#include <algorithm>  
#include <ratio>
#include <unordered_set>
#include "board.h"

using namespace std;

class heuristicEvaluation {
	private:
		int heuristic5(othelloBoard board, int nSpacesRemaining,int symbol);
	public:
		int hIndex = 0;
    	int heuristic(othelloBoard board, int nSpacesRemaining, int symbol);
	    heuristicEvaluation();
};