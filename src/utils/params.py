import argparse

parser = argparse.ArgumentParser()
parser.add_argument('command', type=str, help='command for work flow flag')
parser.add_argument('--gpu', type=str, help='command for number of gpu')
args = parser.parse_args()
