from config import PAIRS
from train_pair import train_pair

for pair in PAIRS.keys():
    train_pair(pair)
