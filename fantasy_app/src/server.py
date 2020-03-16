from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify
import pandas as pd

import json

app = Flask(__name__)

@app.route("/")
def hello(): 
    print('main........')  
    df = pd.read_csv('data/batting_data_1996_2019.csv')
    data = df['Team'].unique() 
    #json_data = df.to_json(orient='records')[1:-1].replace('},{', '} {')
    json_data = json.dumps(data.tolist())
    print(json_data)
    return render_template('index.html', data=json_data)

# run the application
if __name__ == "__main__":
    print('main........')  
    app.run(host='0.0.0.0',debug=True,port=8080)