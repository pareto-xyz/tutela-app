from typing import Any
from gensim.models import Word2Vec


def main(args: Any):
    model: Word2Vec = Word2Vec(
        corpus_file = args.corpus_file,
        vector_size = args.dim,
        workers = args.workers,
        epochs = args.epochs,
        alpha = args.lr,
        seed = args.seed,
    )
    print('saving mode file.')
    model.save(args.model_file)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('corpus_file', type=str, help='path to cached sequences')
    parser.add_argument('model_file', type=str, help='path to save model')
    parser.add_argument('--epochs', type=int, default=5, help='epochs (default: 5)')
    parser.add_argument('--workers', type=int, default=4, help='workers (default: 4)')
    parser.add_argument('--min-count', type=int, default=5, help='min count (default: 5)')
    parser.add_argument('--lr', type=float, default=0.025, help='learning rate (default: 0.025)')
    parser.add_argument('--dim', type=float, default=128, help='dimensionality (default: 128)')
    parser.add_argument('--seed', type=float, default=42, help='random seed (default: 42)')
    args: Any = parser.parse_args()

    main(args)
