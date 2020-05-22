import argparse

parser = argparse.ArgumentParser()

# Parse arguments
parser.add_argument('--label', type=str, help='Label for run')
parser.add_argument('--label', type=str, help='Label for run')

parser.add_argument('-f', type=str, help='Path to kernel json')

# Extract args
ARGS, unknown = parser.parse_known_args()
