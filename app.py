from flask import Flask,render_template,request

import pickle
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    
    return render_template('home.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method == 'POST':
        model = pickle.load(open('model.pkl','rb'))
                
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        
        
        #print(final_features)
        pred_promote = model.predict(final_features)
        #return "promotion"+pred_promote
        return render_template('result.html',prediction = pred_promote)	
    #return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)