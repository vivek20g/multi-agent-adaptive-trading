"""Simple CLI wrapper for simulator data generation."""
import argparse
from . import generate_dataset


def main():
    p = argparse.ArgumentParser(description='Simulator CLI')
    p.add_argument('--excel', default='simulator/data/NiftyPrice.xlsx', help='Input excel with price data')
    p.add_argument('--out', default='simulator/output/simulated_trades.csv', help='Output CSV path')
    p.add_argument('--sheet', default='HDFCBANK', help='Excel sheet name/index')
    args = p.parse_args()

    path = generate_dataset(args.excel, args.out, args.sheet)
    print(f'Generated dataset at: {path}')


if __name__ == '__main__':
    main()
