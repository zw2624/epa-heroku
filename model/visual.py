import bokeh
import pandas as pd
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, Blueprint
import app

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models.widgets import DataTable, TableColumn, HTMLTemplateFormatter
import scipy.special
from bokeh.layouts import gridplot
from bokeh.models.sources import ColumnDataSource
import random
from bokeh.models import Legend


bp = Blueprint('data', __name__, url_prefix='/data')


def create_figure(student_name, epa, data):
    df =  data[(data['student_name'] == student_name) & (data['EPA_pred'] == epa)]
    source = ColumnDataSource(df)
    hover = HoverTool(
        tooltips=[
            ("Date", "$x"),
            ("Score", "$y"),
            ("Comments", "@answer_sentence"),
        ]
    )
    p = figure(width=640, height=480, x_axis_type="datetime", tools=[hover], title = 'Score for:...')
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Score'
    p.line(x='eval_date', y='Sentiment_pred', source=source)
    p.circle(x='eval_date', y='Sentiment_pred', source=source)
    return p



@bp.route('/visualize', methods=('GET', 'POST'))
def visualize():
    predict_data = pd.read_excel('/Users/zwang199/Projects/NLP_Interface_v2/downloads/pred_harry_potter_data.xlsx',
                               encoding='utf-8', sheet_name='Sheet1')
    grouped_data = predict_data.groupby(['student_name', 'EPA_pred', 'eval_date'], as_index=False) \
                               .agg({'answer_sentence': list,
                                     'Sentiment_pred':'mean'})
    all_names = predict_data.student_name.unique()
    all_epas = predict_data.EPA_pred.unique()
    current_student = request.args.get("student_name")
    if current_student == None:
        current_student = all_names[0]
    current_epa = request.args.get("selected_epa")

    if current_epa == None:
        current_epa = all_epas[0]
    plot = create_figure(current_student, current_epa, grouped_data)
    script, div = components(plot)
    return render_template("visualization.html", script=script, div=div,
                           all_names=all_names,
                           all_epas = all_epas,
                           current_student=current_student,
                           current_epa = current_epa)

def create_source_object(df_col, pred_or_actual, bar_color):
  arr_hist, edges = np.histogram(df_col, bins = int(5 / 1), range = [1, 5])
  arr_df = pd.DataFrame({'proportion': arr_hist / np.sum(arr_hist),
                               'left': edges[:-1], 'right': edges[1:] })
  arr_df['f_proportion'] = ['%0.5f' % proportion for proportion in arr_df['proportion']]
  arr_df['name'] = pred_or_actual
  arr_df['color'] = bar_color
  arr_df['ct'] = arr_hist
  src = ColumnDataSource(arr_df)
  return src


def get_tabs():
    epa_data = pd.read_csv("/Users/zwang199/Downloads/NLP_Interface_v2/downloads/wbs_data.csv")
    epa_data.columns = ['EPA', 'Accuracy', 'Precision', 'Recall', 'Count']
    epa_data = pd.DataFrame.from_dict(epa_data)
    source = ColumnDataSource(data=epa_data)
    template = """
    <div style="background:<%= 
        (function colorfromint(){
            if(value < 0.5 && value < 1.1){
                return("#FFCABE")}
            if(value > 0.9 && value < 1.1){
                return("#D1FFBE")}
            }()) %>; 
        color: black"> 
    <%= value %></div>
    """
    formater = HTMLTemplateFormatter(template=template)
    columns = [
        TableColumn(field="EPA", title="EPA"),
        TableColumn(field="Count", title="Count", formatter=formater),
        TableColumn(field="Accuracy", title="Accuracy", formatter=formater),
        TableColumn(field="Precision", title="Precision", formatter=formater),
        TableColumn(field="Recall", title="Recall", formatter=formater)
    ]

    data_table = DataTable(source=source, columns=columns, width=800)
    epas = epa_data.sort_values('Accuracy', ascending=False).iloc[:, 0].values[::-1]
    accuracy = epa_data.sort_values('Accuracy', ascending=False).iloc[:, 1].values[::-1]
    dot = figure(title="EPA Accuracy", tools="", toolbar_location=None, plot_width=400, plot_height=400,
                 y_range=epas, x_range=[0, 1.05], x_axis_label='Accuracy Score', y_axis_label='EPA')

    dot.segment(0, epas, accuracy, epas, line_width=0.5, line_color="#1F5FBB", )
    dot.circle(accuracy, epas, size=10, line_color="#1F5FBB", fill_color="#FFB20D", line_width=0.5)

    sent_pred = pd.DataFrame([random.randint(1, 4) for x in range(5000)])
    sent_actual = pd.DataFrame([random.randint(1, 5) for x in range(5000)])
    sent = pd.concat([sent_pred, sent_actual], axis=1)
    sent.columns = ['sent_pred', 'sent_actual']
    sent.head()

    src_pred = create_source_object(sent['sent_pred'], 'Predicted', 'red')
    src_actual = create_source_object(sent['sent_actual'], 'Actual', 'blue')

    hist = figure(plot_height=400, plot_width=500, title='Sentiment Distribution', x_axis_label='Sentiment Score',
                  y_axis_label='Frequency')

    r1 = hist.quad(source=src_pred, bottom=0, top='proportion', left='left', right='right',
                   color='color', line_color="black", alpha=0.4)
    r2 = hist.quad(source=src_actual, bottom=0, top='proportion', left='left', right='right',
                   color='color', line_color="black", alpha=0.4)

    legend = Legend(items=[
        ("Predicted", [r1]),
        ("Actual", [r2])
    ], location="center")

    hist.add_layout(legend, 'right')
    grid = gridplot([[dot, hist], [data_table]], toolbar_location=None)
    return dot

def get_hist():
    p = figure(width=640, height=480)
    return p

def get_table():
    epa_data = pd.read_csv("/Users/zwang199/Projects/NLP_Interface_v2/downloads/wbs_data.csv")
    epa_data.columns = ['EPA', 'Accuracy', 'Precision', 'Recall', 'Count']
    epa_data = pd.DataFrame.from_dict(epa_data)