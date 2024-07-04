
import os
import sys
import subprocess
import shutil

py_site_pkgs = os.path.join(os.path.dirname(__file__), 'py-site-packages')
os.makedirs(py_site_pkgs, exist_ok=True)
sys.path.append(py_site_pkgs)

try:
  import psutil
except:
  import pip
  print(f'Installing psutil into {py_site_pkgs}')
  pip.main([
    'install', f'--target={py_site_pkgs}', 'psutil'
  ])
  import psutil


def replace_matching_line(file_path, is_match_lambda, replacement_str):
  with open(file_path, 'r') as fd:
    file_contents = fd.read()
  new_file_contents = ''
  for line_num, line in enumerate(file_contents.splitlines(keepends=True)):
    if is_match_lambda(line):
      print(f'Replaced {file_path}:{line_num} original content "{line.strip()}" with new content "{replacement_str.strip()}"')
      new_file_contents += replacement_str
    else:
      new_file_contents += line
  with open(file_path, 'w') as fd:
    fd.write(new_file_contents)


def main():
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

  mirror_gaze_main_cpp = os.path.abspath(os.path.join('src', 'main.cpp'))

  # Add our main.cpp to the build system
  gemma_CMakeLists_txt = os.path.join(gemma_repo_root, 'CMakeLists.txt')
  replace_matching_line(
    gemma_CMakeLists_txt,
    lambda line: 'add_executable' in line and 'gemma/run.cc' in line,
    f'add_executable(gemma {mirror_gaze_main_cpp})\n'
  )

  # update GEMMA_MAX_SEQLEN to be huge
  gemma_configs_h = os.path.join(gemma_repo_root, 'gemma', 'configs.h')
  wanted_seq_len = '32768'
  replace_matching_line(
    gemma_configs_h,
    lambda line: '#define' in line and 'GEMMA_MAX_SEQLEN' in line and not (wanted_seq_len in line),
    f'#define GEMMA_MAX_SEQLEN {wanted_seq_len}\n'
  )

  # Now run the build
  subprocess.run([
    'cmake', '-B', 'build',
  ], cwd=gemma_repo_root, check=True)

  subprocess.run([
    'make', '-j4',
  ], cwd=os.path.join(gemma_repo_root, 'build'), check=True)


  build_dir = os.path.join(
    os.path.abspath(os.getcwd()), 'build'
  )
  os.makedirs(build_dir, exist_ok=True)



  mirror_gaze_exe = os.path.join(build_dir, 'mirror-gaze')
  if sys.platform.startswith('win'):
    mirror_gaze_exe = mirror_gaze_exe+'.exe'

  shutil.copyfile(
    os.path.join(gemma_repo_root, 'build', 'gemma' + ('.exe' if sys.platform.startswith('win') else '') ),
    mirror_gaze_exe
  )

  subprocess.run([
    'chmod', '+x', mirror_gaze_exe,
  ])

  possible_model_file_locations = [
    os.environ.get('GEMMA_MODEL_SBS_FILE', ''),
    '/mnt/scratch/llm-models/google-gemma/7b-it-sfp/7b-it-sfp.sbs',
    '/llm-models/google-gemma/7b-it-sfp/7b-it-sfp.sbs',
  ]

  possible_tokenizer_file_locations = [
    os.environ.get('GEMMA_TOKENIZER_SPM_FILE', ''),
    '/mnt/scratch/llm-models/google-gemma/7b-it-sfp/tokenizer.spm',
    '/llm-models/google-gemma/7b-it-sfp/tokenizer.spm',
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
    print('= = = = = = = = = = = = = = = = = = = = = = = = = = =')

    subproc_env = dict(os.environ)
    subproc_env['GEMMA_MODEL_SBS_FILE'] = model_file
    subproc_env['GEMMA_TOKENIZER_SPM_FILE'] = tokenizer_file

    print(f'GEMMA_MODEL_SBS_FILE={model_file}')
    print(f'GEMMA_TOKENIZER_SPM_FILE={tokenizer_file}')

    sys_argv_idx_of_dash = -1
    for i,arg_val in enumerate(sys.argv):
      if arg_val == '--':
        sys_argv_idx_of_dash = i
        break

    # systemd-run handles making sure this process never gets > 10gb of main system ram, while allowing it to swap like crazy instead of crashing.
    if not (shutil.which('systemd-run') is None):
      ram_size_bytes = psutil.virtual_memory().total
      ram_size_gb = round(ram_size_bytes / (1024 * 1024 * 1024), 2)
      if ram_size_gb < 24:
        # Limit model to 75% of free ram so we don't bring the system to a stuttery mess
        memory_high = int(ram_size_gb * 0.74)
        cmd = [
          'systemd-run', '--scope',
            '-p', f'MemoryHigh={memory_high}G', '-p', 'MemorySwapMax=999G',
            '--nice=18',
          '--user',
          mirror_gaze_exe
        ]
      else:
        # We have plenty of ram to run the model!
        cmd = [ mirror_gaze_exe ]
    else:
      # We can't limit the process anyway, so we won't.
      cmd = [ mirror_gaze_exe ]


    if sys_argv_idx_of_dash >= 0:
      cmd += sys.argv[sys_argv_idx_of_dash+1:]

    print(f'> {" ".join(cmd)}')
    subprocess.run(cmd,
      env=subproc_env,
      check=True
    )



if __name__ == '__main__':
  main()

