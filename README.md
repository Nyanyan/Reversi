# Reversi

## How to start (person vs person)
open ```main.py``` and check that ```ai_mode``` is False.
Then type
```
$ python main.py
```

## How to start (person vs AI)

### Compile

To compile `ai_cpp.cpp`, type

```
$ python compile.py ai_cpp.cpp
```

### Execute

Type

```
$ python main.py
```
then, the following description appears:
```
PERSON: person vs person
BLACK: person(black) vs AI(white)
WHITE: AI(black) vs person(white)
choose: # type your choice
```

## How to play
For each actions, you type a coordinate. For example,
```
  a b c d e f g h 
1 . . . . . . . . 
2 . . . . . . . .
3 . . . * . . . .
4 . . * X O . . .
5 . . . O X * . .
6 . . . . * . . .
7 . . . . . . . .
8 . . . . . . . .
Black:c4
```
Place that you can place is shown as ```*```.
If you place a stone on the right-down corner, you should write ```h8```.

