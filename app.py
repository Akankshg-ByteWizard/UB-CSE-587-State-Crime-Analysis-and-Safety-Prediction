# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd

# Initialize a Flask web application
app = Flask(__name__, template_folder='templates')

# Set a secret key for the Flask session
app.secret_key = 'your_secret_key'

# Load the dataset from a CSV file
project_df = pd.read_csv("Dataset.csv")

# Drop rows with missing values (NaN)
project_df = project_df.dropna()

# Convert data types of columns based on their content
project_df = project_df.convert_dtypes()

# Convert the 'State' column to lowercase for consistency
project_df['State'] = project_df['State'].str.lower()

# Create a copy of the DataFrame for further processing
df = project_df.copy()

# Initialize a label encoder to convert categorical 'State' values to numerical
label_encoder = LabelEncoder()

# Encode the 'State' column to numerical values
df['State'] = label_encoder.fit_transform(df['State'])

# Create a mapping of state names to their encoded values
state_label_mapping = pd.DataFrame({'State': project_df['State'].unique(), 'Label': df['State'].unique()})
state_mapping = dict(zip(state_label_mapping['State'], state_label_mapping['Label']))

# Split the dataset into features (X) and target (y)
X = df[['State', 'Year', 'Data.Population', 'Data.Totals.Property.All', 'Data.Totals.Violent.All']]
y = df['Data.Total.crime']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a list of states
states_list = ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
               'connecticut', 'delaware', 'district of columbia', 'florida', 'georgia', 'hawaii',
               'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
               'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi',
               'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey',
               'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma',
               'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
               'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
               'west virginia', 'wisconsin', 'wyoming']

# Load the trained machine learning model
loaded_model = joblib.load("grid_search.pkl")

# Define a route for the home page
@app.route("/")
def home():
    return render_template('base.html', states_list=states_list)

# Function to make a prediction using the loaded model
def predict_value(to_predict_dict):
    try:
        # Make a prediction using the loaded model
        result = loaded_model.predict(pd.DataFrame([to_predict_dict]))
        return result[0]
    except Exception as e:
        return str(e)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        state_name = request.form['state'].strip().lower()
        
        # Convert the selected state name to the encoded value using the mapping
        state_encoded = state_mapping.get(state_name, -1)
        
        if state_encoded == -1:
            return render_template('result.html', prediction='Invalid state selected.')
        
        year = int(request.form['year'])
        population = int(request.form['population'])
        property_crime = int(request.form['property_crime'])
        violent_crime = int(request.form['violent_crime'])
    except ValueError:
        return render_template('result.html', prediction='Invalid input. Please enter valid values.')

    # Validate user input
    if len(str(year)) != 4 or year < 0:
        return render_template('result.html', prediction='Year must be a four-digit non-negative number.')
    
    if population < 0:
        return render_template('result.html', prediction='Population must be non-negative.')

    if property_crime <= 0 or violent_crime <= 0:
        return render_template('result.html', prediction='Property and Violent Crimes must be greater than zero.')

    if population <= property_crime or population <= violent_crime:
        return render_template('result.html', prediction='Population must be greater than Property Crime or Violent Crime.')

    # Create a dictionary for prediction
    to_predict_dict = {
        'State': state_encoded,
        'Year': year,
        'Data.Population': population,
        'Data.Totals.Property.All': property_crime,
        'Data.Totals.Violent.All': violent_crime
    }

    # Store user input in the session for plotting
    session['user_input'] = to_predict_dict

    # Make a prediction
    prediction = predict_value(to_predict_dict)
    return render_template('result.html', prediction=prediction)

# Define a route for plotting
@app.route('/plot', methods=['GET', 'POST'])
def plot():
    # Retrieve user input from the session
    user_input = session.get('user_input')
    
    if user_input:
        # Create a DataFrame from user input
        user_input_df = pd.DataFrame([user_input])

        # Make a prediction using the loaded model
        prediction = int(loaded_model.predict(user_input_df)[0])

        # Create a scatter plot for user input vs. dataset
        plt.figure(figsize=(8, 6))
        plt.scatter(user_input_df['Year'], prediction, color='red', marker='X', label='User Input')
        plt.title('Crime vs. Year')
        plt.xlabel('Year')
        plt.ylabel('Total Crime')
        plt.legend()

        # Save the scatter plot as an image
        scatter_image_path = 'static/images/scatter_plot.png'
        plt.savefig(scatter_image_path)

        # Create a bar chart for user input vs. dataset
        plt.figure(figsize=(12, 6))
        sns.barplot(x='State', y='Data.Total.crime', data=df)
        plt.xticks(rotation=90)
        plt.bar(user_input_df['State'], prediction, color='red', label='User Input')
        plt.title('Total Crime by State')
        plt.xlabel('State')
        plt.ylabel('Total Crime')
        plt.legend()

        # Save the bar chart as an image
        bar_image_path = 'static/images/bar_chart.png'
        plt.savefig(bar_image_path)

        # Create a pie chart for user input vs. dataset
        crime_types = ['Property Crime', 'Violent Crime', 'Prediction']
        crime_counts = [user_input_df['Data.Totals.Property.All'].sum(), user_input_df['Data.Totals.Violent.All'].sum(), prediction]

        plt.figure(figsize=(8, 6))
        plt.pie(crime_counts, labels=crime_types, autopct='%1.1f%%', startangle=140)
        plt.title('Crime Type Distribution')
        plt.annotate(f'Total Crime Prediction: {prediction}', xy=(.8, 0.9), fontsize=12, color='red', ha='center')

        # Save the pie chart as an image
        pie_chart_image_path = 'static/images/pie_chart.png'
        plt.savefig(pie_chart_image_path)

        return render_template('plot.html', scatter_image_path=scatter_image_path,
                               bar_image_path=bar_image_path,
                               pie_chart_image_path=pie_chart_image_path)

    return render_template('plot.html', scatter_image_path='static/images/scatter_plot.png',
                           bar_image_path='static/images/bar_chart.png',
                           pie_chart_image_path='static/images/pie_chart.png')

# Run the Flask app if this script is executed
if __name__ == '__main__':
    app.run(debug=True)
