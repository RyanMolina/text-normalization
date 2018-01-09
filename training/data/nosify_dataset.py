import os
import random
import time
from multiprocessing import Pool
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.data import load
from utils import generate_vocab as gv
from training.data.TextNoisifier import TextNoisifier


def csv_to_dict(file):
    d = {}
    with open(file, 'r') as f:
        rows = f.read().split('\n')
        for row in rows:
            k, v = row.split(',')
            d.update({k:v})
    return d


accent_dict = csv_to_dict('training/data/common_accented_words.txt')
contract_dict = csv_to_dict('training/data/common_contracted_words.txt')
phonetic_dict = csv_to_dict('training/data/common_phonetically_styled_words.txt')
expansion_dict = {v: k for k, v in contract_dict.items()}

with open('training/data/hyph_fil.tex', 'r') as f:
    hyphenator_dict = f.read()

ntg = TextNoisifier(accent_dict, phonetic_dict, contract_dict, expansion_dict, hyphenator_dict)


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


def collect_dataset(src, tgt, tok=None, max_token_count=140, char_level_emb=False, augment_data=True):

    process_pool = Pool()

    # Instance of PunktSentenceTokenizer from nltk.tokenize module
    if tok:
        tokenizer = load(tok).tokenize
    else:
        tokenizer = sent_tokenize

    print('# Reading file')
    with open(src, encoding='utf8') as infile:
        contents = infile.read()
        articles = [content for content in contents.split('\n') if content != '\n']

    clean_text = []
    if augment_data:
        # add the vocabulary to dataset
        print('  [+] adding all the unique words in char-level embedding to augment the data')
        words, _ = zip(*gv.get_vocab(src, to_file=False))
        clean_text.extend(words)

    print('  [+] converting to list of sentences')
    articles_sentences = process_pool.map(tokenizer, articles)
    print('  [+] flattening the list of sentences')
    sentences = [sentence
                 for sentences in articles_sentences
                 for sentence in sentences]
    clean_text.extend(sentences)
    clean_text_size = len(clean_text)
    print('  [+] Flattened data length: {}'.format(len(clean_text)))

    print('  [+] randomizing the position of the dataset')
    sampled_dataset = random.sample(clean_text, len(clean_text))

    sent_number = 0
    start_time = time.time()

    html_spec_chars = re.compile(r'&#\d+;')

    names = re.compile(r"[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*\s+[A-Z]([a-z]+|\.)")
    acronyms = re.compile(r'\b[A-Z][a-zA-Z\.]*[A-Z]\b\.?')

    path, filename = os.path.split(tgt)
    print('   => collecting clean and noisy sentences')
    with open(tgt, 'w', encoding="utf8") as outfile, \
            open("{}/noisy_{}".format(path, filename), 'w', encoding='utf8') as clean_text:

        for sentence in sampled_dataset:
            sent_number += 1

            if sent_number % 10000 == 0:
                speed = 10000 / (time.time() - start_time)
                print("      # {} "
                      "line/s: {:.2f} "
                      "ETA: {}".format(sent_number, speed, (clean_text_size - sent_number) / speed))
                start_time = time.time()

            sent_len = len(list(sentence))
            words = word_tokenize(sentence)
            if 0 < sent_len < max_token_count \
                    and is_ascii(sentence) \
                    and '‘' not in sentence \
                    and '"' not in sentence \
                    and "'" not in sentence \
                    and not ntg.re_digits.search(sentence) \
                    and not html_spec_chars.search(sentence) \
                    and sentence[0].isalnum():
                # Separate each token with space, even the punctuations
                # Because word_tokenize changes the " to `` | ''
                clean_sentence = ' '.join(words).replace("''", '"').replace("``", '"')

                # Normalize first the contracted words from News Site Articles
                clean_sentence = ntg.expansion(clean_sentence)
                clean_sentence = ntg.expandable_expr.sub(ntg.word_expansion, clean_sentence)

                noisy_sentence = ntg.contraction(clean_sentence)
                noisy_sentence = ntg.contractable_expr.sub(ntg.word_contraction, noisy_sentence)
                noisy_sentence = ntg.anable_expr.sub(ntg.word_ang_to_an, noisy_sentence)
                noisy_sentence = ntg.anu_expr.sub(ntg.word_ano, noisy_sentence)
                noisy_sentence = ntg.amable_expr.sub(ntg.word_ang_to_am, noisy_sentence)
                noisy_sentence = ' '.join(process_pool.map(noisify, noisy_sentence.split()))

                if char_level_emb:
                    clean_sentence = ' '.join(list(clean_sentence)).replace(' ' * 3, ' <space> ')
                    noisy_sentence = ' '.join(list(noisy_sentence)).replace(' ' * 3, ' <space> ')

                if clean_sentence and noisy_sentence:
                    print(clean_sentence, file=outfile)
                    print(noisy_sentence, file=clean_text)

        outfile.truncate(outfile.tell() - 1)
        clean_text.truncate(clean_text.tell() - 1)
