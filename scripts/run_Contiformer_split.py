from joblib import Parallel, delayed
import subprocess

commands = []

for i in range(32, 36):
    commands.append(f'bash scripts/Contiformer_single/Contiformer_{i}.sh')


# 定义一个函数，用于执行Bash命令
def run_bash_command(command):
    subprocess.call(command, shell=True)


# 并行运行Bash命令
num_cores = 16  # 指定并行核心数
Parallel(n_jobs=num_cores, verbose=1)(delayed(run_bash_command)(cmd) for cmd in commands)
