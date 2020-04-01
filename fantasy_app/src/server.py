from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify, Response
from flask_mysqldb import MySQL
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import request
import json, decimal

app = Flask(__name__)

# use with docker image
# app.config['MYSQL_HOST'] = 'fantasy_baseball_db'
# use with localhost
app.config['MYSQL_HOST'] = '127.0.0.1' 
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'dbo'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError

@app.route("/")
def index(): 
    cur = mysql.connection.cursor()
    cur.execute("SELECT DISTINCT Name FROM dbo.BattingDemo")
    data = cur.fetchall()
    return render_template('index.html', data=data)

@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        players = request.form.getlist('checks')
        if len(players) == 2:
            return render_template('data.html', players=players)
        else:
            return ('', 204)
    return render_template('sorting_bat_modify.html')

@app.route("/baseballtest")
def get_home_data():
    cur = mysql.connection.cursor()
    cur.execute("SELECT Season views, Name id, Team created_on, Age title, G, AB, PA, H, `1B`, `2B`, `3B`, HR, R, RBI, BB, IBB, SO, HBP, SF, SH, GDP, SB, CS, AVG, ROW_NUMBER() OVER (ORDER BY Season) num FROM dbo.BattingDemo WHERE Season = 2019 ORDER BY HR DESC")
    data = cur.fetchall()
    response = Response(response=json.dumps(data, default=decimal_default), status=200, mimetype="application/json")
    return(response)

@app.route("/individual/<player_name>")
def render_individual_stats(player_name):
    return render_template('individual_bat.html', player_name=player_name)


@app.route("/stat/<player_name>")
def get_individual_stats(player_name):
    cur = mysql.connection.cursor()
    cur.execute("SELECT Season views, Name id, Team created_on, Age title, G, AB, PA, H, `1B`, `2B`, `3B`, HR, R, RBI, BB, IBB, SO, HBP, SF, SH, GDP, SB, CS, AVG, ROW_NUMBER() OVER (ORDER BY Season DESC) num FROM dbo.BattingDemo WHERE Name = %s ORDER BY Season DESC", (player_name,))
    data = cur.fetchall()
    response = Response(response=json.dumps(data, default=decimal_default), status=200, mimetype="application/json")
    return(response)

# @app.route("/compare")
# def render_individual_stats(player_name):
#     return render_template('individual_bat.html', player_name=player_name)

@app.route("/pandastest")
def test_pandas():
    data = pd.read_sql("SELECT Name, G, HR, AB, PA, H, SB, CS, AVG, R, RBI FROM dbo.BattingDemo", mysql.connection)
    return None


# run the application
if __name__ == "__main__":
    print('main........')  
    app.run(host='0.0.0.0',debug=True,port=8080)