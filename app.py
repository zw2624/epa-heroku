import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from bokeh.embed import components

from model import clean_data, predict_label, visual
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
ALLOWED_EXTENSIONS = {'xls', 'xlsx', 'csv'}

app = Flask(__name__)
app.secret_key = 'some_secret'
app.config.from_object(__name__)
import auth, db
app.register_blueprint(auth.bp)
app.register_blueprint(visual.bp)



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', have_table = False,
                           script = None, div = None)

def process_file(path, filename):
    d_data = clean_data.delimited_data(path)
    predict_data = predict_label.predict_labels(d_data)
    writer = predict_label.pd.ExcelWriter(app.config['DOWNLOAD_FOLDER'] + 'pred_' + filename, engine='xlsxwriter')
    predict_data.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')
    writer.save()
    return

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], 'pred_' + filename, as_attachment=True)


if __name__ == '__main__':
    app.run()
