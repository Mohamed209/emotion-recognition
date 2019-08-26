from flask import Flask
from flask import render_template

app = Flask(__name__,
            template_folder='templates')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict')
def predict():
    pass#return "Salary Predictions : " + str(model.predict(np.array(testdf['xtest']).reshape(-1, 1)))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
