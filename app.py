from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import numpy as np
import pickle
import pandas as pd
import joblib

app = Flask(__name__, template_folder="template")

model_fit = joblib.load(open("./models/RF_fit_model.pkl", "rb"))
print("Model Loaded")
imputer1 = joblib.load(open("./models/imputer.pkl", "rb"))
print("Model Loaded")
scaler = joblib.load(open("./models/scaler.pkl", "rb"))
print("Model Loaded")
encoder = joblib.load(open("./models/encoder.pkl", "rb"))
print("Model Loaded")
input_columns = joblib.load(open("./models/input_columns.pkl", "rb"))
print("Model Loaded")
target_column = joblib.load(open("./models/target_column.pkl", "rb"))
print("Model Loaded")
numeric_columns = joblib.load(open("./models/numeric_columns.pkl", "rb"))
print("Model Loaded")
categorical_columns = joblib.load(open("./models/categorical_columns.pkl", "rb"))
print("Model Loaded")
encoded_columns = joblib.load(open("./models/encoded_columns.pkl", "rb"))
print("Model Loaded")


@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
		
       
        # ['online_order','book_table','votes','location','restaurant_type','cuisines','costof2plates','meal_type','city']

        # Online_order
        onlineOrder = (request.form['onlineorder'])
        # book_table
        bookTable = (request.form['booktable'])
        # restaurant_type
        restaurantType = (request.form['restaurantype'])
        # cuisines
        cuisines = (request.form['cuisines'])
        # meal_type
        mealType = (request.form['mealtype'])
        # location
        location = (request.form['location'])
        # city
        city = (request.form['city'])
        # votes
        votes = float(request.form['votes'])
        # costof2plates
        costOf2Plates = float(request.form['costof2plates'])
        

        input_lst = [onlineOrder , bookTable , restaurantType , cuisines, mealType ,location , city, votes,costOf2Plates]
								
        new_input = {
            'online_order': onlineOrder,
            'book_table': bookTable,
            'restaurant_type': restaurantType,
            'cuisines': cuisines,
            'meal_type': mealType,
            'location': location,
            'city': city,
            'votes': votes,
            'costof2plates': costOf2Plates
        }

        def predict_input(input):
            input_df = pd.DataFrame([input])
            input_df[numeric_columns] = imputer1.transform(input_df[numeric_columns])
            input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
            input_df[encoded_columns] = encoder.transform(input_df[categorical_columns])
            X_input = input_df[numeric_columns + encoded_columns]
            pred = model_fit.predict(X_input)[0]
            return pred
        prediction = predict_input(new_input)
        # prediction = round(prediction, 2)
        output = prediction

        if output>0:
                 return render_template("ratings.html",prediction=prediction ,output1 =round(prediction, 2) )
        else:
                return render_template("predictor.html")
    return render_template("predictor.html")

if __name__=='__main__':
	app.run(debug=True)