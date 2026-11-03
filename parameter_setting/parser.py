import argparse

def parse_in_args():

	parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)

	parser.add_argument('--data_path', default=None, type=str, help='Path to the dataset storage')
	parser.add_argument('--ds_name', default=None, type=str, help='Name of the dataset')
	parser.add_argument('--device', default=5, type=int, help='Target GPU')
	parser.add_argument('--ds_pctg', default=1.0, type=float, help='Dataset %')
	parser.add_argument('--n_jobs', default=4, type=int, help='Dataset %')
	
	
	return parser.parse_args()
