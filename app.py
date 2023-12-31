from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    # int_features.insert(0, 10000)

    # Convert the list to a NumPy array and reshape
    final_features = np.array(int_features).reshape(1, -1)

    # Make predictions
    prediction = model.predict(final_features)
    # print(prediction)

    if prediction == [4]:
        return render_template('index.html',pred='Malignant. It is a problem.')
    else:
        return render_template('index.html',pred='Benign. It is safe.')


if __name__ == '__main__':
    app.run(debug=True)
