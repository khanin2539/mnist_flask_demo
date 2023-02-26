# json conda certu app:  curl -X POST localhost:5000/train -d "{\"arg1\":0.01,\"arg2\":2,\"arg3\":64}" -H "Content-Type:application/json"
# json conda certu app:  curl -X POST localhost:5000/infer -F file=@187.png

import numpy as np
from datetime import datetime
from pytorch_utils import predict, transform_image, retrain
from flask import Flask ,url_for,render_template,request,abort
from flask import request, jsonify
import sqlite3
conn = sqlite3.connect('system_log2.db',check_same_thread=False)
c = conn.cursor()
now = datetime.now()
formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
app = Flask(__name__)

# create a light database
# c.execute("""CREATE TABLE system_log2 (datetime text, endpoint text, status integer)""")
@app.route('/')
def hello():
    return 'Hello World! Lets train and imference'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        # we will get the file from the request
        # handle errors
        try:
            arg = request.get_json()
            # extract the arguments from the json
            arg1 = arg.get('arg1')
            arg2 = arg.get('arg2')
            arg3 = arg.get('arg3')
            data = retrain(arg1, arg2, arg3)
            c.execute("INSERT INTO system_log2(datetime, endpoint, status) VALUES(?,?,?)", (formatted_date, 'train', 200))
            c.execute("SELECT * FROM system_log2")
            print(c.fetchall())
            conn.commit()
            return jsonify(status=200, train_results='training accruacy: {}, validation accuracy: {}'.format(data[0], data[1]))
        # return 500 error to user 
        except:
            c.execute("INSERT INTO system_log2(datetime, endpoint, status) VALUES(?,?,?)", (formatted_date, 'train', 500))
            c.execute("SELECT * FROM system_log2")
            print(c.fetchall())
            conn.commit()
            return jsonify({'error': '500 Internal Server Error'})
            # abort(500)


# @app.errorhandler(500)
# def internal_error(error):

#     return "500 Internal Server Error."


@app.route('/infer', methods=['POST'])
def infer():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
         # handle errors
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            output = predict(tensor)
            c.execute("INSERT INTO system_log2(datetime, endpoint, status) VALUES(?,?,?)", (formatted_date, 'infer', 200))
            c.execute("SELECT * FROM system_log2")
            conn.commit()
            print(c.fetchall())
            return jsonify(status=200, predicted_class= output)
         # return 500 error to user 
        except:
            c.execute("INSERT INTO system_log2(datetime, endpoint, status) VALUES(?,?,?)", (formatted_date, 'infer', 500))
            c.execute("SELECT * FROM system_log2")
            print(c.fetchall())
            conn.commit()
            # abort(500)
            return jsonify({'error': '500 Internal Server Error'})        
app.run()
