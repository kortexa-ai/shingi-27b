"""Run the fixed sequence benchmark against an isolated CUDA runtime.

Requires the instrumented readout from results/sequence-scaling/experimental.patch.
Kernel experiments are selected by the runtime library directory, not a service change.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-path', type=Path, required=True)
    parser.add_argument('--cublas', action='store_true')
    parser.add_argument('--mmq-j', type=int, choices=[32, 64, 128], default=128)
    args, benchmark_args = parser.parse_known_args()
    path = args.library_path.resolve(strict=True)
    if not (path / 'libggml-cuda.so').is_file():
        parser.error('library path must contain libggml-cuda.so')
    os.environ['LD_LIBRARY_PATH'] = str(path)
    os.environ['SHINGI_TRIAL_MMQ_J'] = str(args.mmq_j)
    if args.cublas:
        os.environ['SHINGI_TRIAL_PQ2_CUBLAS'] = '1'
    else:
        os.environ.pop('SHINGI_TRIAL_PQ2_CUBLAS', None)
    sys.argv = ['benchmark_sequence_scaling.py', *benchmark_args]
    runpy.run_path(str(Path(__file__).with_name('benchmark_sequence_scaling.py')), run_name='__main__')


if __name__ == '__main__':
    main()
