


from flask import Flask, render_template, request
from implementation import svm_test, svm_train, svm_predict
import numpy as np

app = Flask(__name__)
app.url_map.strict_slashes = False

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST']) 
def predict():
    data = [float(request.form['value' + str(i)]) for i in range(1, 31)]
    data_np = np.asarray(data, dtype=float).reshape(1, -1) 
    
    out, acc, t = svm_predict(clf, data_np)
    output = 'Malignant' if out == 1 else 'Benign'
    
    acc_x, acc_y = acc[0]
    acc1 = max(acc_x, acc_y)
    return render_template('result.html', output=output, accuracy=acc1, time=t)

if __name__ == '__main__':
    global clf 
    clf = svm_train()
    svm_test(clf)
    app.run(debug=True)

