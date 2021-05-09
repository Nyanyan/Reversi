import sys
import subprocess
import os

def debug(*args): print(*args, file=sys.stderr)

cmd = 'python setup.py build_ext --inplace'
try:
    o = subprocess.run(cmd.split(), input='ai.pyx', encoding='utf-8', stderr=subprocess.STDOUT, timeout=None)
except subprocess.CalledProcessError as e:
    debug('ERROR:', e.stdout)
    exit()

with open('ai_cython_exe.cp38-win_amd64.pyd', 'r') as f:
    pass

debug('------------------compile done------------------')