"""Script to do all the work:
- generate dataset
- train the model
"""
import os
import argparse
import shutil
from training.data import nosify_dataset
from utils import generate_vocab, split_dataset, sent_tokenizer
import seq2seq


def main():
    """Contains all of the work that needs to be done."""
    args = parse_args()
    model_name = args.model_name
    prefix = "char_level"

    corpus_path = os.path.join('training', 'data', 'corpus')
    articles_path = os.path.join(corpus_path, args.src)

    training_path = os.path.join('training', 'data', 'dataset')
    os.makedirs(os.path.join(training_path, model_name), exist_ok=True)

    sent_tokenizer_path = os.path.join(
        'training', 'data', 'sent_tokenizer.pickle')

    if args.train_sent_tokenizer:
        sent_tokenizer.train(articles_path, sent_tokenizer_path)

    if args.generate_dataset:
        nosify_dataset.collect_dataset(articles_path,
                                       os.path.join(
                                           training_path, model_name,
                                           '{}_sentences.txt'.format(prefix)),
                                       tok=sent_tokenizer_path,
                                       char_level_emb=True,
                                       augment_data=True,
                                       shuffle=True,
                                       max_token_count=140*2)

        split_dataset.split(
            os.path.join(
                training_path, model_name,
                'noisy_{}_sentences.txt'.format(prefix)),
            os.path.join(training_path, model_name),
            'enc', test_size=500, dev_size=500)

        split_dataset.split(
            os.path.join(
                training_path, model_name,
                '{}_sentences.txt'.format(prefix)),
            os.path.join(training_path, model_name),
            'dec', test_size=500, dev_size=500)

        generate_vocab.get_vocab(
            os.path.join(training_path, model_name, 'train.enc'))
        generate_vocab.get_vocab(
            os.path.join(training_path, model_name, 'train.dec'))

    if args.train:
        data_dir = os.path.join(training_path, model_name)
        model_dir = os.path.join('training', 'model', model_name)
        os.makedirs(model_dir, exist_ok=True)
        hparams_file = os.path.join('{}_hparams.json'.format(prefix))

        if not os.path.exists(os.path.join(model_dir, hparams_file)):
            shutil.copy(hparams_file, os.path.join(model_dir))

        hparams = seq2seq.utils.load_hparams(
            os.path.join(model_dir, hparams_file))
        trainer = seq2seq.trainer.Trainer(
            data_dir=data_dir, model_dir=model_dir, hparams=hparams)
        trainer.train()


def parse_args():
    """Parse the arguments needed before running.
    Returns:
        args : contains the source text, model name, and if
               should generate dataset or train.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help="Filename of your source text")
    parser.add_argument('--model_name', default='model', type=str,
                        help="Unique name for each model you train.")

    parser.add_argument('--generate_dataset', default=False, type=bool,
                        help="Generate parallel noisy text")

    parser.add_argument('--train', default=False, type=bool,
                        help="Start/Resume train")

    parser.add_argument('--train_sent_tokenizer', default=False, type=bool,
                        help="Train a new sentence tokenizer")

    return parser.parse_args()


if __name__ == "__main__":
    main()
