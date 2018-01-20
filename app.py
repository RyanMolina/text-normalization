"""
This module handles the page routing and rendering.
It also handles the REST API for the text normalization model.
"""
import os
import argparse
import tensorflow as tf
from flask import (Flask, render_template, jsonify, Response,
                   stream_with_context, Markup, request)
import serve

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
APP = Flask(__name__)


@APP.context_processor
def highlight_incorrect():
    """Returns highlighted text if doesn't match"""
    def _compare(tgt, res):
        output = []
        for i, e in enumerate(res):
            try:
                if e != tgt[i]:
                    output.append('<span style="color: red;">{}</span>'
                                  .format(e))
                else:
                    output.append(e)
            except IndexError:
                pass
        return Markup(' '.join(output))
    return dict(highlight_incorrect=_compare)


@APP.route('/')
def index():
    """Returns the index.html"""
    return render_template('index.html')


@APP.route('/normalize/test', methods=['POST'])
def accuracy_test():
    """Normalizing the test files on the fly.
    The trick is to have an inner function that uses a
    generator to generate data and to invoke that function
    and pass it to a Response object.
    """
    enc = request.form['enc-data']
    dec = request.form['dec-data']

    def stream_template(template_name, **context):
        """Returns stream object from stream()
        instead of string from render()

        Args:
            template_name (str): filename of the template
            **context (tuple): keyword arguments

        Returns:
            StreamObject: stream object of the template
        """
        APP.update_template_context(context)
        template = APP.jinja_env.get_template(template_name)
        stream = template.stream(context)
        stream.enable_buffering(5)
        return stream

    def generate():
        """Returns a generator for Lazy-loading of table rows
        Yields:
            dict: The input, expected, and system output
        """
        if '<space>' in enc:
            enc_content = enc.replace(' ', '') \
                             .replace('<space>', ' ').split('\n')
            dec_content = dec.replace(' ', '') \
                             .replace('<space>', ' ').split('\n')
        else:
            enc_content = enc.split('\n')
            dec_content = dec.split('\n')

        for i, e in enumerate(enc_content[:1000]):
            if e:
                result = {
                          'enc': e.strip().strip('\n'),
                          'dec': dec_content[i].strip().strip('\n').split(),
                          'res': NORMALIZER.model_api(
                              e.strip().strip('\n')).split()
                          }
                yield result

    return Response(stream_with_context(
        stream_template('accuracy_testing.html', rows=generate())))


@APP.route('/normalize/api', methods=['POST'])
def normalize():
    """The route for REST API which returns the normalized input.
    Decorators:
        APP
    Returns:
        json: contains the input and the normalized output
    """
    src = request.form['src']
    output = NORMALIZER.model_api(src)
    return jsonify({'src': src, 'tgt': output})


@APP.errorhandler(404)
def url_error(error):
    """Returns the error message for wrong URL.
    Decorators:
        APP
    Arguments:
        error: error message
    Returns:
        string: HTML formatted error message.
    """

    return """
    Wrong URL!
    <pre>{}</pre>""".format(error), 404


@APP.errorhandler(500)
def server_error(error):
    """Returns the error message for Internal server error.
    Decorators:
        APP
    Arguments:
        error: error message
    Returns:
        string: HTML formatted error message.
    """

    return """
    An internal error occured: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(error), 500


def parse_args():
    """Parse the arguments needed before running.
    Returns:
        args: contains the parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="Dir of your selected model and the checkpoint.")

    parser.add_argument('--model_name', default='model_served', type=str,
                        help="""
                        Name of the model to use.
                        Change only if you want to try other models.
                        (Default: 'model_served')
                        """)
    parser.add_argument('--checkpoint', default=None, type=str,
                        help="""
                        Specify the checkpoint filename.
                        (Default: latest checkpoint)
                        """)
    parser.add_argument('--char_emb', default=False, type=bool,
                        help="""
                        Char-level or word-level embedding
                        """
                        )
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    with tf.Session() as sess:
        NORMALIZER = serve.Serve(sess=sess,
                                 model_name=ARGS.model_name,
                                 checkpoint=ARGS.checkpoint,
                                 char_emb=ARGS.char_emb)
        APP.run(debug=True, use_reloader=True)
