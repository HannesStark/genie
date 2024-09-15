
import argparse
from tqdm import tqdm
import numpy as np
from genie.sampler.unconditional import UnconditionalSampler
from genie.utils.multiprocessor import MultiProcessor
from genie.utils.model_io import load_pretrained_model

def main(args):

    model = load_pretrained_model(
        args.rootdir,
        args.name,
        args.epoch,
    ).eval().cuda()

    # Load sampler
    sampler = UnconditionalSampler(model)

    lens = np.load(args.len_dist)['lengths']
    
    sample_lens = []
    for i in range(args.num_batches):
        choice = 0
        while choice < 4 or choice > 500:
            choice = np.random.choice(lens)
        sample_lens.append(choice)
    if args.sorted:
        sample_lens = sorted(sample_lens)[::-1]
    
    # Iterate through all tasks
    for i, length in tqdm(enumerate(sample_lens)):
        print(f'{i} of {len(sample_lens)}')
        batch_size = args.batch_size

        params = {
            'length': length,
            'scale': args.scale,
            'num_samples': batch_size,
            'outdir': args.outdir,
            'prefix': args.run_name + str(length),
            'offset': i
        }
        sampler.sample(params)
        

if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser()

    # Define model arguments
    parser.add_argument('--name', type=str, help='Model name', required=True)
    parser.add_argument('--epoch', type=int, help='Model epoch', required=True)
    parser.add_argument('--rootdir', type=str, help='Root directory', default='results')
    parser.add_argument('--run_name', type=str, help='Root directory', default='')

    # Define sampling arguments
    parser.add_argument('--scale', type=float, help='Sampling noise scale', required=True)
    parser.add_argument('--outdir', type=str, help='Output directory', required=True)
    parser.add_argument('--num_samples', type=int, help='Number of samples per length', default=5)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--min_length', type=int, help='Minimum sequence length', default=50)
    parser.add_argument('--max_length', type=int, help='Maximum sequence length', default=256)
    parser.add_argument('--length_step', type=int, help='Length step size', default=1)
    parser.add_argument('--num_batches', type=int, help='Length step size', default=1)
    parser.add_argument('--len_dist', type=str, help='Output directory', default=None)
    
    # Define environment arguments
    parser.add_argument('--num_devices', type=int, help='Number of GPU devices', default=1)
    parser.add_argument('--sorted', action='store_true', help='Run in decreasing order of length')

    # Parse arguments
    args = parser.parse_args()

    # Run
    main(args)