# Reversi

## How to start (person vs person)
open ```main.py``` and check that ```ai_mode``` is False.
Then type
```
$ python main.py
```


## How to start (person vs AI)
open ```main.py``` and check that ```ai_mode``` is True.
```ai_player``` is ```1``` when AI is white, ```0``` when AI is black.
Then type
```
$ python main.py
```

## How to play
For each actions, you type a coordinate. For example,
```
  0 1 2 3 4 5 6 7
00●●○○○○○○
10●○●○○○○○
20●○○●○○○○
30●○○●●○○○
40●○●○○●○○
50●●○○●○○○
60●●●●○*..
70○○●●●●●*
Black: 77
```
Place that you can place is shown as ```*```.
If you place a stone on the right-down corner, you should write ```77```.
