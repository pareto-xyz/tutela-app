import os
from typing import Any
from src.diff2vec.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class TrainCallback(CallbackAny2Vec):
    """
    Save model every epoch and log epoch completion.
    """

    def __init__(self, save_dir: str):
        self._save_dir: str = save_dir
        self._epoch: int = 0

    def on_train_begin(self, _):
        print('training start')

    def on_train_end(self, _):
        print('done')

    def on_epoch_begin(self, _):
        print(f'epoch {self._epoch} start')
        self._epoch += 1

    def on_epoch_end(self, model):
        out_path: str = os.path.join(self._save_dir, f'word2vec-epoch{self._epoch}.model')
        print(f'epoch {self._epoch} end')
        model.save(out_path)
        print(f'saved model to {out_path}')


def main(args: Any):
    model: Word2Vec = Word2Vec(
        corpus_file = args.corpus_file,
        vector_size = args.dim,
        workers = args.workers,
        epochs = args.epochs,
        alpha = args.lr,
        seed = args.seed,
        corpus_size = 263644512,
        callbacks = [TrainCallback(args.model_dir)],
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('corpus_file', type=str, help='path to cached sequences')
    parser.add_argument('model_dir', type=str, help='path to save model')
    parser.add_argument('--epochs', type=int, default=5, help='epochs (default: 5)')
    parser.add_argument('--workers', type=int, default=4, help='workers (default: 4)')
    parser.add_argument('--min-count', type=int, default=5, help='min count (default: 5)')
    parser.add_argument('--lr', type=float, default=0.025, help='learning rate (default: 0.025)')
    parser.add_argument('--dim', type=float, default=128, help='dimensionality (default: 128)')
    parser.add_argument('--seed', type=float, default=42, help='random seed (default: 42)')
    args: Any = parser.parse_args()

    main(args)
