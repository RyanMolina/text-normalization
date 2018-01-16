import os
import random
import time
from multiprocessing import Pool
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.data import load
from utils import generate_vocab as gv
from training.data.textnoisifier import TextNoisifier


def csv_to_dict(file):
    d = {}
    with open(file, 'r') as f:
        rows = f.read().split('\n')
        for row in rows:
            k, v = row.split(',')
            d.update({k: v})
    return d


accent_dict = csv_to_dict('training/data/common_accented_words.txt')
contract_dict = csv_to_dict('training/data/common_contracted_words.txt')
phonetic_dict = \
    csv_to_dict('training/data/common_phonetically_styled_words.txt')
expansion_dict = {v: k for k, v in contract_dict.items()}

with open('training/data/hyph_fil.tex', 'r') as f:
    hyphenator_dict = f.read()

ntg = TextNoisifier(accent_dict, phonetic_dict, contract_dict,
                    expansion_dict, hyphenator_dict)


def is_ascii(text):
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def noisify(text):
    """
    function wrapper to be fed on Pool.map()
    :param text:
    :return noisy text:
    """
    return ntg.noisify(text)


def collect_dataset(src, tgt, tok=None, max_token_count=50,
                    char_level_emb=False, augment_data=False,
                    shuffle=False, size=None):

    process_pool = Pool()

    # Instance of PunktSentenceTokenizer from nltk.tokenize module
    if tok:
        tokenizer = load(tok).tokenize
    else:
        tokenizer = sent_tokenize

    print('# Reading file')
    with open(src, encoding='utf8') as infile:
        contents = infile.read()
        articles = [content
                    for content in contents.split('\n') if content != '\n']

    dataset = []
    if augment_data:
        # add the vocabulary to dataset
        print("  [+] adding all the unique words in "
              "char-level embedding to augment the data")
        words, _ = zip(*gv.get_vocab(src, to_file=False))
        dataset.extend(words)

    print('  [+] converting to list of sentences')
    articles_sentences = process_pool.map(tokenizer, articles)
    print('  [+] flattening the list of sentences')
    sentences = [sentence
                 for sentences in articles_sentences
                 for sentence in sentences]
    dataset.extend(sentences)
    dataset_size = len(dataset)
    print('  [+] Flattened data length: {}'.format(dataset_size))

    if size:
        dataset = dataset[:size]

    if shuffle:
        print('  [+] randomizing the position of the dataset')
        dataset = random.sample(dataset, dataset_size)

    sent_number = 0
    start_time = time.time()

    print('   => collecting clean and noisy sentences')
    path, filename = os.path.split(tgt)
    with open(tgt, 'w', encoding="utf8") as decoder_file, \
            open("{}/noisy_{}".format(path, filename),
                 'w', encoding='utf8') as encoder_file:

        for sentence in dataset:
            sent_number += 1

            if sent_number % 10000 == 0:
                speed = 10000 / (time.time() - start_time)

                print("      # {} "
                      "line/s: {:.2f} "
                      "ETA: {}".format(
                          sent_number,
                          speed,
                          (dataset_size - sent_number) / speed))

                start_time = time.time()

            sent_len = len(list(sentence))
            words = word_tokenize(sentence)
            if 0 < sent_len < max_token_count \
                    and is_ascii(sentence):
                # Separate each token with space, even the punctuations
                # Because word_tokenize changes the " to `` | ''
                clean_sentence = ' '.join(words) \
                                    .replace("''", '"') \
                                    .replace("``", '"')

                # Normalize first the contracted words from News Site Articles
                clean_sentence = ntg.expansion(clean_sentence)
                clean_sentence = ntg.expandable_expr.sub(ntg.word_expansion,
                                                         clean_sentence)

                noisy_sentence = clean_sentence

                noisy_sentence = ntg.contraction(noisy_sentence)

                if random.getrandbits(1):
                    noisy_sentence = ntg.contractable_expr.sub(
                        ntg.word_contraction, noisy_sentence)

                if random.getrandbits(1):
                    noisy_sentence = ntg.anable_expr.sub(
                        ntg.word_ang_to_an, noisy_sentence)

                if random.getrandbits(1):
                    noisy_sentence = ntg.anu_expr.sub(
                        ntg.word_ano, noisy_sentence)

                if random.getrandbits(1):
                    noisy_sentence = ntg.amable_expr.sub(
                        ntg.word_ang_to_am, noisy_sentence)

                if random.getrandbits(1):
                    noisy_sentence = ntg.remove_space_expr.sub(
                        ntg.word_remove_space, noisy_sentence)

                if random.getrandbits(1):
                    noisy_sentence = ' '.join(process_pool.map(
                        noisify, noisy_sentence.split()))

                if random.getrandbits(1):
                    if random.getrandbits(1):
                        noisy_sentence = noisy_sentence.lower()
                    else:
                        noisy_sentence = noisy_sentence.upper()
                else:
                    noisy_sentence = noisy_sentence.title()

                if char_level_emb:
                    clean_sentence = ' '.join(list(clean_sentence)) \
                                        .replace(' ' * 3, ' <space> ')
                    noisy_sentence = ' '.join(list(noisy_sentence)) \
                                        .replace(' ' * 3, ' <space> ')

                if clean_sentence and noisy_sentence:
                    print(clean_sentence, file=decoder_file)
                    print(noisy_sentence, file=encoder_file)

        decoder_file.truncate(decoder_file.tell() - 1)
        encoder_file.truncate(encoder_file.tell() - 1)
