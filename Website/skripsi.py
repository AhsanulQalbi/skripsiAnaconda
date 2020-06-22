from flask import Flask, redirect, url_for, render_template, request, session
from datetime import timedelta
from sklearn.feature_extraction.text import  TfidfVectorizer
from flask import send_file, flash

import numpy as numpy
import pandas as pd
import pickle
import re
import string
import nltk
import dill 
import sklearn
from joblib import dump, load

app = Flask(__name__, static_folder='static')
app.secret_key = "hello"
app.permanent_session_lifetime = timedelta(minutes=3)
model = load('RF_seleksi.joblib') 
# model_train_features = pd.read_csv('Rf_seleksi_features.csv').features
vectorizer = load('vectroizer_seleksi.joblib')


@app.route("/textPrediction", methods = ["POST", "GET"])
def home() :
    sentiment = None
    konten = None
    prediksi = None
    if request.method == "POST" :
        sentiment = request.form["sentiment"]
        #preprocessing
        sentiment = sentiment.lower()
        sentiment = re.sub(r"\d+", "",sentiment)
        sentiment = sentiment.translate(str.maketrans('','', string.punctuation))
        sentiment = " ".join(re.findall("[a-zA-Z]+", sentiment))
        konten = sentiment
        sentiment = [sentiment]
        Data_test= vectorizer.transform(sentiment)

        prediksi = model.predict(Data_test)
        if prediksi == [1]:
            prediksi = 'Positive'
        else :
            prediksi = 'Negative'

    return render_template("indeks.html", konten = konten, prediksi = prediksi)

@app.route("/", methods = ["POST", "GET"])
def filePrediction() :
    sentiment_data = None
    prediksi_result = None
    total_data = None
    data_view = []
    positivePercentage = None
    negativePercentage = None
    prediksi = None

    if request.method == "POST" :
        if request.form['download'] == "0":
            file = request.files.get('fileUpload') 
            if file :
                if file.content_type != 'application/vnd.ms-excel' :
                    return redirect(url_for('filePrediction'))
                
                uploaded_data = request.files['fileUpload']

                sentiment_data = pd.read_csv(uploaded_data)
                column_name = "``".join([str(i) for i in sentiment_data.columns.tolist()])

                clean_data = sentiment_data[column_name].astype(str)    
                clean_data = clean_data.apply(lambda data: data.lower()) #Lower Case
                clean_data = clean_data.apply(lambda data: re.sub(r"\d", "", data)) #Remove Number    
                clean_data = clean_data.apply(lambda data: data.translate(str.maketrans('','',string.punctuation))) #punctuation  
                clean_data = clean_data.apply(lambda data: " ".join(re.findall("[a-zA-Z]+", data)))

                Data_test = vectorizer.transform(clean_data).toarray()
                prediksi = model.predict(Data_test)
                prediksi_result = {}  
                for indeks, value in enumerate(prediksi):
                    if value == 1:
                        prediksi_result[indeks] = 'Positive'
                    else :
                        prediksi_result[indeks] = 'Negative'
                
                data_view = sentiment_data[column_name]

                totalData = len(prediksi_result)
                total_positive = 0
                total_negative = 0

                for x in prediksi:
                    if x == 1:
                        total_positive = total_positive + 1
                    else : 
                        total_negative = total_negative + 1

                
                positivePercentage = (total_positive/totalData)*100
                negativePercentage = (total_negative/totalData)*100
            

                list_prediksi = list(prediksi_result.values())
                data_DF = list(zip(data_view, list_prediksi))
                df = pd.DataFrame(data_DF, columns=["sentiment","prediction"])
                df.to_csv('ResultData.csv', index=False) 

            else :
                return redirect(url_for('filePrediction'))

        if request.form['download'] == "1":
            return send_file('ResultData.csv',
                     mimetype='text/csv',
                     attachment_filename='ResultData.csv',
                     as_attachment=True)

    return render_template("fileInput.html", data = data_view, prediksi = prediksi_result, length = len(data_view), positive = positivePercentage, negative = negativePercentage)
                     

@app.route("/download", methods = ["POST", "GET"])
def filedownload() :

    return render_template("fileInput.html", data = data_view, prediksi = prediksi_result, length = len(data_view))


@app.route("/login", methods = ["POST", "GET"])
def login():
    if request.method == "POST" :
        session.permanent = True
        user = request.form["nm"]
        session["user"] = user
        return redirect(url_for("user"))
    return render_template("login.html")

@app.route("/about")
def user():
    return render_template("about.html")


@app.route("/logout")
def logout():
    session.pop("user",None)
    session.pop("email",None)
    return redirect(url_for("login"))

    
if __name__ == "__main__" :
    app.run(debug=True)