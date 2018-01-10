import os
import tensorflow as tf
from flask import (Flask, render_template, jsonify, Response, url_for,
                   stream_with_context, Markup, request, redirect)
import serve

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
app = Flask(__name__)

@app.context_processor
def highlight_incorrect():
    def _compare(tgt, res):
        output = []
        for i, e in enumerate(res):
            try:
                if e != tgt[i]:
                    output.append('<span style="color: red;">{}</span>'.format(e))
                else:
                    output.append(e)
            except IndexError:
                pass
        return Markup(' '.join(output))
    return dict(highlight_incorrect=_compare)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/normalize/test', methods=['POST'])
def accuracy_test():
    enc = request.form['enc-data']
    dec = request.form['dec-data']

    def stream_template(template_name, **context):
        app.update_template_context(context)
        t = app.jinja_env.get_template(template_name)
        rv = t.stream(context)
        rv.enable_buffering(5)
        return rv

    def generate():
        enc_content = enc.replace(' ', '').replace('<space>', ' ').split('\n')
        dec_content = dec.replace(' ', '').replace('<space>', ' ').split('\n')
        for i, e in enumerate(enc_content[:1000]):
            if e:
                result = {'enc': e.strip().strip('\n'),
                          'dec': dec_content[i].strip().strip('\n').split(),
                          'res': normalizer.model_api(e.strip().strip('\n')).split()}
                yield result

    return Response(stream_with_context(
                            stream_template('accuracy_testing.html',
                                            rows=generate())))


@app.route('/normalize/api', methods=['POST'])
def normalize():
    src = request.form['src']
    output = normalizer.model_api(src)
    return jsonify({'src': src, 'tgt': output})


@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occured: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    with tf.Session() as sess:
        normalizer = serve.Serve(sess=sess, model_name="model_bicol",
                                 checkpoint="normalize.ckpt-47000")
        app.run(debug=True, use_reloader=True)
