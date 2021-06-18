import sys
import subprocess
import os

argv = sys.argv

cmd = 'python setup.py build_ext --inplace'
try:
    o = subprocess.run(cmd.split(), input=argv[1], encoding='utf-8', stderr=subprocess.STDOUT, timeout=None)
except subprocess.CalledProcessError as e:
    print('ERROR:', e.stdout)
    exit()

with open('ai_cython_exe.cp38-win_amd64.pyd', 'r') as f:
    pass

print('------------------compile done------------------')