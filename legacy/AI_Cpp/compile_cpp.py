import sys
import subprocess
def debug(*args): print(*args, file=sys.stderr)

cmd = 'g++ ai_cpp.cpp -O3 -march=native -fexcess-precision=fast -funroll-loops -flto -mtune=native -o a.exe'
o = subprocess.run(cmd.split(), encoding='utf-8', stderr=subprocess.STDOUT, timeout=None)
debug('------------------compile done------------------')