# Reversi

## How to start (person vs person)
open ```main.py``` and check that ```ai_mode``` is False.
Then type
```
$ python main.py
```

## How to start (person vs AI)

### Create parameters

#### Create your own parameters

To compile ```ai_fast.cpp```, type

```
$ python compile.py ai_fast.cpp
```

Then create parameters with executing

```
python adjust_params.py
```

This program runs forever. To stop it, please use ```CTRL+C```.

```param.txt``` is made.

#### Use default parameters

Rename ```param_base.txt``` to ```param.txt```.

### Compile

To compile `ai_cpp.cpp`, type

```
$ python compile.py ai_cpp.cpp
```

### Execute

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

