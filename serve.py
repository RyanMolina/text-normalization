"""Contains the Serve class"""
import os
from nltk.tokenize import word_tokenize, sent_tokenize
import seq2seq


class Serve:
    """Serve an instance of the trained model"""
<<<<<<< HEAD
    def __init__(self, sess, model_name, checkpoint, char_emb=False):
        os.makedirs(os.path.join('training', 'data', 'dataset', model_name),
                    exist_ok=True)

        data_dir = os.path.join('training', 'data', 'dataset', model_name)
        model_dir = os.path.join('training', 'model', model_name)

        hparams = seq2seq.utils.load_hparams(
            os.path.join(model_dir, 'hparams.json'))

        self.char_emb = char_emb
=======
    def __init__(self, sess, model_name, checkpoint):
        os.makedirs(os.path.join('training', 'data', 'dataset', model_name),
                    exist_ok=True)
        data_dir = os.path.join('training', 'data', 'dataset', model_name)
        model_dir = os.path.join('training', 'model', model_name)
        hparams = seq2seq.utils.load_hparams(
            os.path.join(model_dir, 'char_level_hparams.json'))
>>>>>>> 453c72b83592310389b1b478246326275740ef9b
        self.normalizer = seq2seq.predictor.Predictor(sess,
                                                      dataset_dir=data_dir,
                                                      output_dir=model_dir,
                                                      output_file=checkpoint,
                                                      hparams=hparams)

    def model_api(self, input_data):
        """Does all the preprocessing before prediction.
        1. Split into sentences
        2. Tokenize to words
        3. Tokenize into characters
        4. encode ' ' into <space>
        Args:
            input_data (string): informal text
        Returns:
            string: normalized text
        """

        output = ""
        for sentence in sent_tokenize(input_data):
            if not self.char_emb:
                tokens = ' '.join(word_tokenize(sentence))
            else:
                tokens = self._char_emb_format(sentence)

            normalized = self.normalizer.predict(tokens)
            output += normalized.replace(' ', '').replace('<space>', ' ')
        return output

    @staticmethod
    def _char_emb_format(text):
        return ' '.join(list(text)).replace(' ' * 3, ' <space> ')
