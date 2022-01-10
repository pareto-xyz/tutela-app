import os
import sys
import psycopg2
import numpy as np
import pandas as pd
from os.path import join
from typing import Tuple, Optional, List, Dict, Any


def main(args: Any):
    pass


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', action='store_true', default=False)
    parser.add_argument('--no-db', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
