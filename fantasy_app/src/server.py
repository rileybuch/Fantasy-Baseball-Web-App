from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify
from flask_mysqldb import MySQL

import json

app = Flask(__name__)

app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'dbo'

mysql = MySQL(app)

@app.route("/")
def index(): 
    cur = mysql.connection.cursor()
    cur.execute("SELECT DISTINCT Name FROM dbo.BattingDemo")
    data = cur.fetchall()
    data_return = [entry[0] for entry in data]
    return render_template('index.html', data=data_return)

@app.route("/<player_name>")
def get_player_stats(player_name):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM dbo.BattingDemo WHERE Name = %s ORDER BY Season", (player_name,))
    num_fields = len(cur.description)
    field_names = [i[0] for i in cur.description]
    player_data = cur.fetchall()
    return render_template('data.html', data=player_data, field_names=field_names)

# run the application
if __name__ == "__main__":
    print('main........')  
    app.run(host='0.0.0.0',debug=True,port=8080)