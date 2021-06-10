import sys
import subprocess

argv = sys.argv

if len(argv) < 2:
    print('arg err')
    exit()

cmd = 'g++ ' + argv[1] + ' -O3 -march=native -fexcess-precision=fast -funroll-loops -flto -mtune=native -o a.exe'
o = subprocess.run(cmd.split(), encoding='utf-8', stderr=subprocess.STDOUT, timeout=None)
print('------------------compile done------------------')