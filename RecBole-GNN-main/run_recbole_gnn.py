import argparse

from recbole_gnn.quick_start import run_recbole_gnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='svecf', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='dataset/lastfm', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='config/lastfm/SVECF.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
