from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
data = pd.read_csv("student_exam_data.csv")

# Encode categorical column
le = LabelEncoder()
data['Extra_Coaching'] = le.fit_transform(data['Extra_Coaching'])

# Features and target
X = data[['Study_Hours_per_Day',
          'Attendance_Percentage',
          'Previous_Score',
          'Sleep_Hours',
          'Assignments_Completed',
          'Extra_Coaching',
          'Internet_Usage_Hours']]

y_score = data['Final_Exam_Score']
y_pass = data['Final_Exam_Score'].apply(lambda x: 1 if x >= 50 else 0)


# Split
X_train, X_test, y_train_score, y_test_score = train_test_split(X, y_score, test_size=0.2, random_state=42)
_, _, y_train_pass, y_test_pass = train_test_split(X, y_pass, test_size=0.2, random_state=42)

# Models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train_score)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train_pass)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.form['Study_Hours_per_Day'])
    attendance = float(request.form['Attendance_Percentage'])
    previous = float(request.form['Previous_Score'])
    sleep = float(request.form['Sleep_Hours'])
    assignments = int(request.form['Assignments_Completed'])
    coaching = request.form['Extra_Coaching']
    internet = float(request.form['Internet_Usage_Hours'])

    coaching_encoded = le.transform([coaching])[0]

    input_data = [[study_hours, attendance, previous, sleep,
                   assignments, coaching_encoded, internet]]

    predicted_score = linear_model.predict(input_data)[0]
    pass_fail = logistic_model.predict(input_data)[0]

    result = "Pass ✅" if pass_fail == 1 else "Fail ❌"

    return render_template("index.html",
                           score=round(predicted_score, 2),
                           result=result,
                           form_data=request.form)


if __name__ == '__main__':
    app.run(debug=True)
