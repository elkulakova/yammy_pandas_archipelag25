from argparse import *
from typing import Tuple

def input_directories() -> Tuple[str, str]:
    parser = ArgumentParser(description='paths to images and result')
    # пока что вариант для ситуации, когда в командную строку вводятся директории без ключей --input_dir/--output_dir
    parser.add_argument('indir', type=str, help='input dir with images')
    parser.add_argument('outdir', type=str, help='output dir to save the result')
    args = parser.parse_args()
    return args.indir, args.outdir

if __name__ == "__main__":
    in_dir, out_dir = input_directories()