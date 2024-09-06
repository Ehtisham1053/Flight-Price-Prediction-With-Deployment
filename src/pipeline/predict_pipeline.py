import pickle
import pandas as pd

# Load preprocessor, model, and label encoders
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
model = pickle.load(open("artifacts/model.pkl", "rb"))
label_encoders = pickle.load(open("artifacts/label_encoders.pkl", "rb"))

def preprocess_and_predict(data):
    # Convert input data to DataFrame
    df = pd.DataFrame([data])

    # Date and Time feature extraction
    df['Date'] = df['Date_of_Journey'].str.split('/').str[0].astype(int)
    df['Month'] = df['Date_of_Journey'].str.split('/').str[1].astype(int)
    df['Year'] = df['Date_of_Journey'].str.split('/').str[2].astype(int)
    df.drop('Date_of_Journey', axis=1, inplace=True)

    df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]
    df['Arrival_Hour'] = df['Arrival_Time'].str.split(':').str[0].astype(int)
    df['Arrival_Min'] = df['Arrival_Time'].str.split(':').str[1].astype(int)
    df.drop('Arrival_Time', axis=1, inplace=True)

    df['Dept_hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
    df['Dept_min'] = df['Dep_Time'].str.split(':').str[1].astype(int)
    df.drop('Dep_Time', axis=1, inplace=True)

    # Mapping 'Total_Stops'
    df['Total_Stops'] = df['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
    df.drop('Route', axis=1, inplace=True)

    # Duration conversion to minutes
    df['Duration_hour'] = df['Duration'].str.split(' ').str[0].str.replace('h', '').astype(int)
    df['Duration_min'] = df['Duration'].str.split(' ').str[1].str.replace('m', '').fillna(0).astype(int)
    df['Duration_in_min'] = df['Duration_hour'] * 60 + df['Duration_min']
    df.drop(['Duration', 'Duration_hour', 'Duration_min'], axis=1, inplace=True)

    # Categorical encoding using stored LabelEncoders
    categorical_features = ['Airline', 'Source', 'Destination', 'Additional_Info']
    for feature in categorical_features:
        le = label_encoders[feature]
        df[feature] = le.transform(df[feature])

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # Transform using the preprocessor
    processed_data = preprocessor.transform(df)

    # Predict using the model
    prediction = model.predict(processed_data)

    return prediction[0]

# Example usage:
# data = {
#     'Date_of_Journey': '24/03/2022',
#     'Arrival_Time': '19:00',
#     'Dep_Time': '08:00',
#     'Total_Stops': 'non-stop',
#     'Route': 'DEL → BOM',
#     'Duration': '2h 50m',
#     'Airline': 'Jet Airways',
#     'Source': 'Delhi',
#     'Destination': 'Cochin',
#     'Additional_Info': 'No info'
# }

# result = preprocess_and_predict(data)
# print(f"Predicted Price: {result}")
