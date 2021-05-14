# Reversi

## How to start (person vs person)
open ```main.py``` and check that ```ai_mode``` is False.
Then type
```
$ python main.py
```

## How to start (person vs AI)

## Compile

To compile `ai_cpp.cpp`, type

```
$ python compile_cpp.py
```

## Execute

open ```main.py``` and check that ```ai_mode``` is True.
```ai_player``` is ```1``` when AI is white, ```0``` when AI is black.
Then type

```
$ python main.py
```

## How to play
For each actions, you type a coordinate. For example,
```
 abcdefgh
1●●○○○○○○
2●○●○○○○○
3●○○●○○○○
4○○●●○○○
5○●○○●○○
6●○○●○○○
7●●●○*..
8○●●●●●*
Black: h8
```
Place that you can place is shown as ```*```.
If you place a stone on the right-down corner, you should write ```h8```.

