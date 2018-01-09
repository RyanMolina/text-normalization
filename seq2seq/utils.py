import json
import codecs
import tensorflow as tf
import numpy as np
import _locale
import sys
import math
import collections
import os

import time

_locale.getdefaultlocale = (lambda *args: 'utf-8')


def check_tensorflow_version():
    min_tf_version = "1.4.0"
    if tf.__version__ < min_tf_version:
        raise EnvironmentError("TensorFlow version must be >= {}".format(min_tf_version))


def load_hparams(hparams_file):
    if tf.gfile.Exists(hparams_file):
        print("+ Loading hparams from {} ...".format(hparams_file))
        with codecs.getreader('utf-8')(tf.gfile.GFile(hparams_file, 'rb')) as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
                return hparams
            except ValueError:
                print('- Error loading hparams file.')
    else:
        raise FileNotFoundError(hparams_file + " not found.")


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""
    print("{}, time {}s, {}.".format(s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def print_hparams(hparams, skip_patterns=None):
    """Print hparams, can skip keys based on patterns."""
    values = hparams.values()
    for key in sorted(values.keys()):
        if not skip_patterns or all(
                [skip_pattern not in key for skip_pattern in skip_patterns]):
            print_out("  %s=%s" % (key, str(values[key])))


def save_hparams(out_dir, hparams):
    hparams_file = os.path.join(out_dir, "hparams")
    print_out("  saving hparams to {}".format(hparams_file))
    with open(hparams_file, 'w') as outfile:
        json.dump(hparams.to_json(), outfile)


def format_text(words):
    """Convert a sequence words into sentence."""
    if (not hasattr(words, "__len__") and  # for numpy array
            not isinstance(words, collections.Iterable)):
        words = [words]
    return b" ".join(words).decode('utf-8')


def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
    Useful to log things that are not part of the train graph, e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def unk_replace(source_tokens,
                predicted_tokens,
                attention_scores):
    result = []
    for token, scores in zip(predicted_tokens, attention_scores):
        if token == b"<unk>":
            max_score_index = np.argmax(scores)
            chosen_source_token = source_tokens[max_score_index]
            new_target = chosen_source_token
            result.append(new_target)
        else:
            result.append(token)
    return result
