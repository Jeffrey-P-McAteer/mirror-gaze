
import os
import sys
import subprocess

os.chdir(
  os.path.dirname(__file__)
)

os.makedirs('build_3rdparty', exist_ok=True)

gemma_repo_root = os.path.join('build_3rdparty', 'gemma.cpp')
if not os.path.exists(gemma_repo_root):
  print(f'Cloning https://github.com/google/gemma.cpp.git to {gemma_repo_root}')
  subprocess.run([
    'git', 'clone', 'https://github.com/google/gemma.cpp.git'
  ], cwd='build_3rdparty', check=True)




