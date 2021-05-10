#include <list>
#include <unordered_map>

using namespace std;

class openings {
	public:	
		list<list<int>> generateData(int symbol);
    	pair<bool,pair<int,list<int>>> getMove(unordered_map<int, list<int>> validMoves ,list<int> pastMoves);
		list<list<int>> sequences;

    	openings();
};