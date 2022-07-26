import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100, help='Number of transactions in a graph')
    parser.add_argument('--relation_path', type=str, default='../../BitcoinData/address_relation/' , help='The path of address_relation of raw dataset')
    parser.add_argument('--label_path', type=str, default='../../BitcoinData/label/', help='The path of lable of raw dataset')
    parser.add_argument('--tx_path', type=str, default='../../BitcoinData/tx/', help='The path of tranction of raw dataset')
    parser.add_argument('--dataset_path', type=str, default='dataset/', help='The path of tranction of raw dataset')
    parser.add_argument('--theta', type=float, default=0.2, help='Threshold of similarity')

    args = parser.parse_args()
    return args