from typing import Any, List
from src.utils.utils import from_pickle
from src.diff2vec.diff2vec import Diff2Vec


def main(args: Any):
    sequences: List[List[int]] = from_pickle(args.sequences_file)
    model: Diff2Vec = Diff2Vec(
        dimensions = args.dim,
        window_size = args.window,
        cover_size = args.cover,
        epochs = args.epochs,
        learning_rate = args.lr,
        workers = args.workers,
        seed = args.seed,
        cache_dir = args.cache_dir,
    )
    model.fit(sequences)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('sequences_file', type=str, help='path to cached sequences')
    parser.add_argument('--epochs', type=int, default=10, help='epochs (default: 10)')
    parser.add_argument('--workers', type=int, default=4, help='workers (default: 4)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate (default: 0.05)')
    parser.add_argument('--dim', type=float, default=128, help='dimensionality (default: 128)')
    parser.add_argument('--window', type=float, default=10, help='window (default: 10)')
    parser.add_argument('--cover', type=float, default=80, help='cover (default: 80)')
    parser.add_argument('--seed', type=float, default=42, help='random seed (default: 42)')
    args: Any = parser.parse_args()

    main(args)
