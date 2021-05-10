#include "openings.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

openings::openings() {
}

list<list<int>> openings::generateData(int symbol) {
	char fName[18];
	sprintf(fName,"%s","lib/openings.csv");
	int nLines = 620;

	ifstream database (fName,ifstream::in);
	char line[1024];
	list<list<int>> seq;

	for (int j = 1; j < nLines; j++) {
		database.getline(line, 1024);
		string str(line);
		std::stringstream ss(str);
		int i;

		list<int> moves;

		while (ss >> i) {
			moves.push_back(i);

			if (ss.peek() == ',')
				ss.ignore();
		}
		seq.push_back(moves);
	}

	return seq;
}

pair<bool, pair<int,list<int>>> openings::getMove(unordered_map<int, list<int>> validMoves, list<int> pastMoves) {
	pair<int,list<int>> move;	
	bool found = false;
	pair<bool, pair<int,list<int>>> moveStatus;
	unordered_map<int, list<int>>::const_iterator candPtr;
	pair<int, list<int>> candidate;
	for (auto & sequence : sequences ) {
		if (sequence.size() > pastMoves.size()) {
			bool stillEqual = true;
			list<int>::const_iterator it1 = pastMoves.begin();
			list<int>::const_iterator it2 = sequence.begin();
			for (; it1 != pastMoves.end() && it2 != sequence.end(); ++it1, ++it2) {
				if (*it1 != *it2) {
					stillEqual = false;
				}
			}
			if (stillEqual) {
				int pos = *it2++;
				for (auto candidate : validMoves) {
					if (candidate.first == pos) {
						move = candidate;
						found = true;
						break;
					}
				}
			}
		}
		sequence.pop_front();
		sequence.pop_front();
		if (found)
			break;
	}

	moveStatus.first = found;
	moveStatus.second = move;

	return moveStatus;
}