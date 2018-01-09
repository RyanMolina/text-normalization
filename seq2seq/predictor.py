import os
import tensorflow as tf
from seq2seq.dataset import Dataset
from seq2seq.model import ModelBuilder
from seq2seq import utils


class Predictor(object):
    def __init__(self, session, dataset_dir, output_dir, output_file, hparams=None):
        self.session = session

        print("+ Preparing dataset placeholder and hyperparameters...")
        self.dataset = Dataset(dataset_dir=dataset_dir, training=False, hparams=hparams)
        self.hparams = self.dataset.hparams

        self.src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        src_dataset = tf.data.Dataset.from_tensor_slices(self.src_placeholder)

        self.batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        self.infer_batch = self.dataset.get_inference_batch(src_dataset, self.batch_size_placeholder)

        print("+ Creating inference seq2seq...")
        self.model = ModelBuilder(training=False,
                                  dataset=self.dataset,
                                  model_dir=output_dir,
                                  batch_input=self.infer_batch)

        print("+ Restoring seq2seq weights ...")
        self.model.saver.restore(session, os.path.join(output_dir, output_file))
        self.session.run(tf.tables_initializer())

    def predict(self, sentence):
        tokens = [token.encode('utf-8') for token in sentence.split()]
        tmp = [b' '.join(tokens).strip()]
        self.session.run(self.infer_batch.initializer,
                         feed_dict={
                            self.src_placeholder: tmp,
                            self.batch_size_placeholder: 1
                         })
        outputs, infer_summary = self.model.infer(self.session)
        outputs = outputs.tolist()[0]

        eos_token = self.hparams.eos_token.encode('utf-8')
        if eos_token in outputs:
            outputs = outputs[:outputs.index(eos_token)]
        out_sentence = utils.format_text(utils.unk_replace(tokens, outputs, infer_summary))
        return out_sentence



