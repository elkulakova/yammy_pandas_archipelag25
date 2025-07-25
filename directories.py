from argparse import *

parser = ArgumentParser(description='paths to images and result')
# пока что вариант для ситуации, когда в командную строку вводятся директории без ключей --input_dir/--output_dir
parser.add_argument('indir', type=str, help='input dir with images')
parser.add_argument('outdir', type=str, help='output dir to save the result')
args = parser.parse_args()