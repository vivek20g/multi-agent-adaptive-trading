"""main.py — top-level orchestrator that composes simulator and trainer packages."""

import argparse
from simulator import generate_dataset
from trainer.train_from_file import train_from_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generate', action='store_true')
    p.add_argument('--train', action='store_true')
    p.add_argument('--excel', default='simulator/data/NiftyPriceHistory.xlsx')
    p.add_argument('--sheet', default='HDFCBANK')
    p.add_argument('--out', default='simulator/output/simulated_trades.xlsx')
    args = p.parse_args()
    print('args:',args)
    data_path = args.out

    if args.generate:
        print('Generating simulated dataset...')
        generate_dataset(args.excel, data_path, args.sheet)

    if args.train:
        print('Training model from', data_path)
        train_from_file(data_path)


if __name__ == '__main__':
    main()


