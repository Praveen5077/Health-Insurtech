import numpy as np
from flask import Flask,request,render_template
import joblib
import sklearn
from forex_python.converter import CurrencyRates

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/main")
def main():
    return render_template("prediction.html")




@app.route("/predict", methods=["GET", "POST"])
def predict():
    age=request.form["age"]
    sex=request.form["gender"]
    bmi=request.form["bmi"]
    child=request.form["child"]
    smoke=request.form["smoke"]
    disease=request.form["disease"]
    region=request.form["region"]
    lst=list([age,sex,bmi,child,smoke,region,disease])
    print(lst)
    features=np.array(lst).reshape(1,-1)
    #age	sex	bmi	children	smoker	region	diseases
    c= CurrencyRates()
    rate=c.get_rate('USD', 'INR')
    print(rate)
    model=joblib.load('rf.joblib')
    print("fea")
    results=model.predict(features)
    print("af fea")
    result="Your predicted amount: Rs."+str(results[0]*rate)
    print("result")
    return render_template('prediction.html', output=result)
if __name__ == "__main__":
    app.run()