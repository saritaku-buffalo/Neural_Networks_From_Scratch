

import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess your CSV data (assuming you have a 'data.csv' file)
data = pd.read_csv('data.csv')
data.fillna(data.median(numeric_only=True).round(1), inplace=True)
data.fillna('Unknown', inplace=True)

# Define features (X) and target (y)
selected_features = ['Job Criteria', 'Job Type', 'Experience Level', 'Country', 'Skills']
X = data[selected_features]
y = data['Salary']

# Preprocessing pipelines for numerical and categorical features
num_features = []
cat_features = ['Job Criteria', 'Job Type', 'Experience Level', 'Country', 'Skills']

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

X_transformed = full_pipeline.fit_transform(X)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_transformed, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_criteria = ['SAP']
        job_type = ['Full Time']
        experience_level = ['Mid Level']
        country = ['United States']
        skills = request.form['Skills']

        new_job_listing = pd.DataFrame({
            'Job Criteria': job_criteria,
            'Job Type': job_type,
            'Experience Level': experience_level,
            'Country': country,
            'Skills': [skills]
        })

        new_job_listing_transformed = full_pipeline.transform(new_job_listing)
        predicted_salary = model.predict(new_job_listing_transformed)

        return render_template('index.html', predicted_salary=predicted_salary[0])
    
    return render_template('index.html', predicted_salary=None)

if __name__ == '__main__':
    app.run(debug=True)



