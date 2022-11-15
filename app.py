import pandas as pd
import random
from sklearn import LogisticRegression
from flask import Flask, request, render_template

app = Flask(__name__)

def encode_categorical(candidates_df):
    gender_mapper = {"M":1, "F":0}
    ethnicity_mapper = {"Black":0, "White":1, "Arab":2, "Asian":3, "Other":4}
    candidates_df["gender"] = candidates_df["gender"].replace(gender_mapper)
    candidates_df["ethnicity"] = candidates_df["ethnicity"].replace(ethnicity_mapper)
    return candidates_df

def calculate_bias(candidates_data):
    candidates_df = pd.DataFrame(candidates_data, columns=['gender','age', 'years_of_experience','skill_score', 'work_ethic','attitude_score', 'ethnicity', 'disability', 'legal_history','country_of_origin', 'time_spent_viewing', 'shortlisted' ]) 
    candidates_df = encode_categorical(candidates_df)
    columns = ['gender','age', 'years_of_experience','skill_score', 'work_ethic','attitude_score', 'ethnicity', 'disability', 'legal_history','country_of_origin', 'time_spent_viewing']
    X = candidates_df[columns]
    y = candidates_df['shortlisted']
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X,y)
    gender_bias = lr.coef_[0]*100
    ethnicity_bias = lr.coef[6]  *100
    nationality_bias = lr.coef_[9] *100
    return gender_bias, ethnicity_bias, nationality_bias
 
@app.route('/calculate_bias', methods = ['POST'])
def bias():
    if request.method == 'POST':
        candidates_data = request.get_json()['data']
        
        gender_bias, ethnicity_bias, nationality_bias = calculate_bias(candidates_data)
                     
        return {'gender bias:': gender_bias, 'ethnicity bias:': ethnicity_bias, 'nationality bias:': nationality_bias}

if __name__ == "__main__":
    app.run(debug=True)