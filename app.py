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
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("file and allowed")
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("uploaded saved")
            # process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
            #return visual.visualize()
            p1 = visual.get_tabs()
            p2 = visual.get_hist()
            script, div = components(p1)
            script2, div2 = components(p2)
            print(script)
            return render_template('index.html', have_table=True,
                                   script=script, div=div,
                                   script2=script2, div2=div2)
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
