from flask import Flask,render_template,request
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle as pk
import pandas as pd

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/submit',methods=['POST','GET'])
def collectData():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chestpain = int(request.form['chestpain'])
        sugar = int(request.form['fbs'])
        exc = int(request.form['exercise'])
        exng = int(request.form['exng'])
        bp = int(request.form['bp'])
        chol = int(request.form['chol'])
        hr = int(request.form['max_heartrate'])
        old = float(request.form['oldpeak'])
        
        if age >= 0 and age <= 14:
            age=0
        elif age >=15 and age<=24:
            age=1
        elif age >=25 and age<=64:
            age=2
        else:
            age=3
            
        data = pd.read_csv(r'heart.csv',usecols=['trtbps','chol','max_heartrate','oldpeak'])
        data.iloc[0,:] = [bp,chol,hr,old]
        num_array = np.array([bp,chol,hr,old])
        scalar = MinMaxScaler()
        num_ss = scalar.fit_transform(data)
        final_data = list(num_ss[0])
        cat_data = [age,sex,chestpain,sugar,exc,exng]
        for i in cat_data:
            final_data.append(i)
            
            
        array = np.array(final_data)
        array = array.reshape(1,-1)
        
        
        model = pk.load(open('model_rf.pkl','rb'))
        prediction = model.predict(array)
        
        if(prediction[0] == 1):
            output = "There is high chance of heart attack......Please do visit the Doctor"
        else:
            output = "No Worries....Take Good care of ur Health"
        
        
        return render_template('output.html', pred=output)



if __name__=='__main__':
    app.run()