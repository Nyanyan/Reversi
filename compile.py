import sys
import subprocess

argv = sys.argv

if len(argv) < 2:
    print('arg err')
    exit()

out_name = 'a.exe' if len(argv) == 2 else argv[2]

cmd = 'g++ ' + argv[1] + ' -O3 -march=native -fexcess-precision=fast -funroll-loops -flto -mtune=native -o ' + argv[2]
o = subprocess.run(cmd.split(), encoding='utf-8', stderr=subprocess.STDOUT, timeout=None)
print('------------------compile done------------------')