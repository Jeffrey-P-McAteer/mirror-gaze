
import os
import sys
import subprocess
import shutil

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

build_dir = os.path.join(
  os.path.abspath(os.getcwd()), 'build'
)
os.makedirs(build_dir, exist_ok=True)



mirror_gaze_exe = os.path.join(build_dir, 'mirror-gaze')
if sys.platform.startswith('win'):
  mirror_gaze_exe = mirror_gaze_exe+'.exe'



possible_model_file_locations = [
  '/mnt/scratch/llm-models/google-gemma/7b-it-sfp/7b-it-sfp.sbs',
  os.environ.get('GEMMA_MODEL_SBS_FILE', '')
]

possible_tokenizer_file_locations = [
  '/mnt/scratch/llm-models/google-gemma/7b-it-sfp/tokenizer.spm',
  os.environ.get('GEMMA_TOKENIZER_SPM_FILE', '')
]

if 'run' in sys.argv:
  print(f'Running {mirror_gaze_exe}')
  model_file = next(iter([x for x in possible_model_file_locations if len(x) > 1 and os.path.exists(x)]), None)
  if model_file is None or not os.path.exists(model_file):
    print('Error, no model files found!')
    print('Set the GEMMA_MODEL_SBS_FILE= environment variable to the path of a c++-variant model')
    print('from https://www.kaggle.com/models/google/gemma/gemmaCpp, such as 7b-it-sfp.sbs')
    sys.exit(1)
  print(f'Using model file {model_file}')

  tokenizer_file = next(iter([x for x in possible_tokenizer_file_locations if len(x) > 1 and os.path.exists(x)]), None)
  if tokenizer_file is None or not os.path.exists(tokenizer_file):
    print('Error, no tokenizer files found!')
    print('Set the GEMMA_TOKENIZER_SPM_FILE= environment variable to the path of a c++-variant tokenizer')
    print('from https://www.kaggle.com/models/google/gemma/gemmaCpp, such as tokenizer.spm')
    print('Ensure the tokenizer matches the model file!')
    sys.exit(1)
  print(f'Using tokenizer file {tokenizer_file}')









