"""
    Data obtained with accuracy level 4 for parameters [1, 0]
"""
import os.path as op

from ..util import load_list

DATA_FILE = op.join(op.dirname(op.abspath(__file__)), "erikkson_4_1_0.txt")

data = load_list(DATA_FILE)
