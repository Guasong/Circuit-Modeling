import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--name', type=str, default='Contiformer')
args = parser.parse_args()

counter = 0

files = os.listdir(args.indir)
files = sorted(files)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)


def output(buffer):
    global counter
    with open(args.outdir + f'{args.name}_{counter}.sh', 'w') as f:
        # f.write(f'export CUDA_VISIBLE_DEVICES={counter % 16}\n\n')
        f.write('\n'.join(buffer))
        f.close()
    counter += 1


for file in files:
    buffer = []
    with open(args.indir + file, 'r') as f:
        for line in f.readlines():
            if 'python' in line:
                if len(buffer):
                    output(buffer)
                buffer = [line.strip()]
            else:
                if len(buffer):
                    buffer.append(line.strip())

    if len(buffer):
        output(buffer)
