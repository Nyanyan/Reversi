CC = g++
CFLAGS = -O3 -std=c++11 -march=native

all: othello

othello: player.o board.o game.o openings.o heuristicEvaluation.o
	${CC} ${CFLAGS} -o ../othello othello.cpp player.o board.o game.o openings.o heuristicEvaluation.o

game.o: game.h game.cpp player.o
	${CC} ${CFLAGS} -c game.cpp

board.o: board.h board.cpp
	${CC} ${CFLAGS} -c board.cpp

player.o: player.h player.cpp
	${CC} ${CFLAGS} -c player.cpp 

openings.o: openings.h openings.cpp
	${CC} ${CFLAGS} -c openings.cpp 

heuristicEvaluation.o: heuristicEvaluation.h heuristicEvaluation.cpp
	${CC} ${CFLAGS} -c heuristicEvaluation.cpp

clean:
	rm ../othello *.o


