from training.data.textnoisifier import TextNoisifier
from multiprocessing import Pool


def noisify(text):
    """
    function wrapper to be fed on Pool.map()
    :param text:
    :return noisy text:
    """
    return ntg.noisify(text)


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
phonetic_dict = csv_to_dict(
    'training/data/common_phonetically_styled_words.txt')
expansion_dict = {v: k for k, v in contract_dict.items()}

with open('training/data/hyph_fil.tex', 'r') as f:
    hyphenator_dict = f.read()

ntg = TextNoisifier(accent_dict,
                    phonetic_dict,
                    contract_dict,
                    expansion_dict,
                    hyphenator_dict)


def main():
    clean_sentence = """Aalis ka ba pag ikaw ang bida ako ang bida.
    Di ko na kaya pang umasa sa wala , ang diyan oo na mahal na kung mahal kita
    , ganoon pala yun gusto ko ayaw ko at gusto ko bahala kayo"""

    clean_sentence = ntg.expansion(clean_sentence)
    clean_sentence = ntg.expandable_expr.sub(ntg.word_expansion,
                                             clean_sentence)
    print(clean_sentence)
    noisy_sentence = ntg.contraction(clean_sentence)

    noisy_sentence = ntg.contractable_expr.sub(
        ntg.word_contraction, noisy_sentence)

    noisy_sentence = ntg.anable_expr.sub(
        ntg.word_ang_to_an, noisy_sentence)

    noisy_sentence = ntg.anu_expr.sub(
        ntg.word_ano, noisy_sentence)

    noisy_sentence = ntg.amable_expr.sub(
        ntg.word_ang_to_am, noisy_sentence)

    noisy_sentence = ntg.remove_space_expr.sub(
        ntg.word_remove_space, noisy_sentence)

    p = Pool()

    noisy_sentence = ' '.join(p.map(
        noisify, noisy_sentence.split()))

    print(noisy_sentence)


if __name__ == '__main__':
    main()
