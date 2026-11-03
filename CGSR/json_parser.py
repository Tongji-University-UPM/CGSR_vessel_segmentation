import json
import argparse

def parse_in_args():

	parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)

	parser.add_argument('--model_type', default=None, type=str, help='Sensor transform network')
	parser.add_argument('--batch_size', default=None, type=int, help='Batch size')
	parser.add_argument('--lr', default=None, type=float, help='Batch size')
	parser.add_argument('--epochs', default=None, type=int, help='Total epochs to run')
	parser.add_argument('--device', default=None, help='device to use for training / testing')
	parser.add_argument('--seed', default=None, type=int)
	parser.add_argument('--skip_train', action='store_true')
	parser.add_argument('--dataset_pctg', default=None, type=float)
	parser.add_argument('--resize', default=None, type=int)
	parser.add_argument('--model_name', default=None, type=str)
	parser.add_argument('--pretrained_model', action='store_true')
	parser.add_argument('--save_chkpt', action='store_true')
	parser.add_argument('--fill_pl_blanks', action='store_true')
	parser.add_argument('--pretrained_model_name', default=None, type=str)
	parser.add_argument('--only_test', action='store_true')
	parser.add_argument('--config_type', default='f', type=str)
	
	
	parser.add_argument('--config', default=None, type=str)

	return parser.parse_args()


def parse_json_args(args):
	
	in_path = args.config

	if args.config_type=="s":
		json_dict = json.loads(in_path)

	elif args.config_type=="f":
		json_dict = json.load(open(in_path))
	else:
		raise ValueError(f'"{args.config_type}" type of config file not recognized.')
		
		
	for k in json_dict.keys():

		try:
			argval = getattr(args,k)

			if isinstance(argval,bool):
				setattr(args, k, json_dict[k] or argval)

			if argval is None:
				setattr(args, k, json_dict[k])
		
		except AttributeError:
			setattr(args, k, json_dict[k])
		
	return args
