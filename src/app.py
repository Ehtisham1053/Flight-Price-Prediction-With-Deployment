from flask import Flask, request, jsonify, render_template
# from src.pipeline import preprocess_and_predict
from src.pipeline.predict_pipeline import preprocess_and_predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = {
            "Airline": request.form['Airline'],
            "Date_of_Journey": request.form['Date_of_Journey'],
            "Source": request.form['Source'],
            "Destination": request.form['Destination'],
            "Route": request.form['Route'],
            "Dep_Time": request.form['Dep_Time'],
            "Arrival_Time": request.form['Arrival_Time'],
            "Duration": request.form['Duration'],
            "Total_Stops": request.form['Total_Stops'],
            "Additional_Info": request.form['Additional_Info']
        }

        prediction = preprocess_and_predict(data)
        return render_template('index.html', prediction_text=f"Predicted Price: {prediction}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
