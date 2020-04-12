from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify, Response
from flask_mysqldb import MySQL
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json, decimal
import io
import base64

app = Flask(__name__)
app.secret_key = "abc" 

# use with docker image
app.config['MYSQL_HOST'] = 'fantasy_baseball_db'
# use with localhost
#app.config['MYSQL_HOST'] = '127.0.0.1' 
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'dbo'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route("/batters", methods=['GET', 'POST'])
def batters():
    session['player_type'] = 'Bat'
    stats = ['G', 'HR', 'AB', 'PA', 'H', 'SB', 'CS', 'AVG', 'R', 'RBI']
    if request.method == 'POST':
        players = request.form.getlist('checks')
        if len(players) == 2:
            session['player_list'] = players
            cur = mysql.connection.cursor()
            cur.execute("SELECT a.Season FROM (SELECT DISTINCT Season Season FROM dbo.Batting WHERE Name = %s) a INNER JOIN (SELECT DISTINCT Season Season FROM dbo.Batting WHERE Name = %s) b ON a.Season = b.Season ORDER BY a.Season DESC", (players[0], players[1],))
            data = cur.fetchall()
            seasons = []
            for entry in data:
                seasons.append(entry['Season'])
            return render_template('data.html', players=players, seasons=seasons, stats= stats)
        else:
            return ('', 204)
    return render_template('sorting_bat_modify.html')

@app.route("/pitchers", methods=['GET', 'POST'])
def pitchers():
    session['player_type'] = 'Pitch'
    stats = ['W', 'L', 'ERA', 'WAR', 'G', 'GS', 'CG', 'ShO', 'SV', 'BS']
    if request.method == 'POST':
        players = request.form.getlist('checks')
        if len(players) == 2:
            session['player_list'] = players
            cur = mysql.connection.cursor()
            cur.execute("SELECT a.Season FROM (SELECT DISTINCT Season Season FROM dbo.Pitching WHERE Name = %s) a INNER JOIN (SELECT DISTINCT Season Season FROM dbo.Pitching WHERE Name = %s) b ON a.Season = b.Season ORDER BY a.Season DESC", (players[0], players[1],))
            data = cur.fetchall()
            seasons = []
            for entry in data:
                seasons.append(entry['Season'])
            return render_template('data.html', players=players, seasons=seasons, stats=stats)
        else:
            return ('', 204)
    return render_template('sorting_pitch_modify.html')

@app.route("/battingdata")
def get_bat_data():
    cur = mysql.connection.cursor()
    cur.execute("SELECT Season views, Name id, Team created_on, Age title, G, AB, PA, H, `1B`, `2B`, `3B`, HR, R, RBI, BB, IBB, SO, HBP, SF, SH, GDP, SB, CS, AVG, ROW_NUMBER() OVER (ORDER BY Season) num FROM dbo.Batting WHERE Season = 2019 ORDER BY HR DESC")
    data = cur.fetchall()
    response = Response(response=json.dumps(data, default=decimal_default), status=200, mimetype="application/json")
    return(response)

@app.route("/pitchingdata")
def get_pitch_data():
    cur = mysql.connection.cursor()
    cur.execute("SELECT Season views, Name id, Team created_on, Age title, W,L,ERA,WAR,G,GS,CG,ShO,SV,BS,IP,TBF,H,R,ER,HR,BB,Balls,Strikes,Pitches, ROW_NUMBER() OVER (ORDER BY Season) num FROM dbo.Pitching WHERE Season = 2019 ORDER BY W DESC")
    data = cur.fetchall()
    response = Response(response=json.dumps(data, default=decimal_default), status=200, mimetype="application/json")
    return(response)

@app.route("/individual_bat/<player_name>")
def render_individual_bat_stats(player_name):
    return render_template('individual_bat.html', player_name=player_name)

@app.route("/individual_pitch/<player_name>")
def render_individual_pitch_stats(player_name):
    return render_template('individual_pitch.html', player_name=player_name)


@app.route("/stat/<player_name>")
def get_individual_stats(player_name):
    cur = mysql.connection.cursor()
    print(session['player_type'])
    if session['player_type'] == 'Bat':
        cur.execute("SELECT Season views, Name id, Team created_on, Age title, G, AB, PA, H, `1B`, `2B`, `3B`, HR, R, RBI, BB, IBB, SO, HBP, SF, SH, GDP, SB, CS, AVG, ROW_NUMBER() OVER (ORDER BY Season DESC) num FROM dbo.Batting WHERE Name = %s ORDER BY Season DESC", (player_name,))
    else:
        cur.execute("SELECT Season views, Name id, Team created_on, Age title, W,L,ERA,WAR,G,GS,CG,ShO,SV,BS,IP,TBF,H,R,ER,HR,BB,Balls,Strikes,Pitches, ROW_NUMBER() OVER (ORDER BY Season) num FROM dbo.Pitching WHERE Name = %s ORDER BY Season DESC", (player_name,))
    data = cur.fetchall()
    response = Response(response=json.dumps(data, default=decimal_default), status=200, mimetype="application/json")
    return(response)

@app.route("/compare", methods=['POST'])
def compare_stats():
    stats = request.form.getlist('stats')
    season = request.form.get('season')
    if len(stats) > 0:
        fig = make_chart(int(season), session['player_list'], stats, session['player_type'])
        encoded = fig_to_base64(fig)
        encoded = encoded.decode('utf-8')
        return render_template('index.html', image=encoded)
    else:
        return ('', 204)

def autolabel(rects, player_list, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    i = 0
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(player_list[i]),
                    xy=(rect.get_x() + rect.get_width() / 2, height - 1),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        i+= 1

def recalculation(one, two):

    one_list = []
    two_list = []

    for i in range(len(one)):

        maxval = 0

        if (one[i] > two[i]):
            maxval = one[i]
        else:
            maxval = two[i]

        const = 100 / maxval
        new_one = int(one[i] * const)
        new_two = two[i] * const

        one_list.append(new_one)
        two_list.append(new_two)

    return one_list, two_list

def make_chart(Year, players, comparison_labels, player_type):
    matplotlib.pyplot.switch_backend('Agg')
    one = []
    two = []

    player_first = players[0]
    player_second = players[1]

    col = ["Season", "Name"]
    for item in comparison_labels:
        col.append(item)
    col.append("")
    if player_type == 'Bat':
        df = pd.read_sql("SELECT Season, Name, G, HR, AB, PA, H, SB, CS, AVG, R, RBI FROM dbo.Batting WHERE Name IN %s", mysql.connection, params=[tuple(players)])
    else:
        df = pd.read_sql("SELECT Season, Name, W, L, ERA, WAR, G, GS, CG, ShO, SV, BS FROM dbo.Pitching WHERE Name IN %s", mysql.connection, params=[tuple(players)])

    for i in range(len(df)):
        if (df['Name'][i] == player_first and df['Season'][i] == Year):
            for label in comparison_labels:
                if (label == 'AVG'):
                    one.append(round(df[label][i], 3))
                else: one.append(df[label][i])
            # one.append(df['G'][i])
            # one.append(df['HR'][i])
            # one.append(df['SB'][i])
            # one.append(df['H'][i])
            # one.append(df['AVG'][i])
            # one.append(df['R'][i])
            # print(df['id'][i]) 

    for i in range(len(df)):
        if (df['Name'][i] == player_second and df['Season'][i] == Year):
            for label in comparison_labels:
                if (label == 'AVG'):
                    two.append(round(df[label][i], 3))
                else: two.append(df[label][i])

    player_one, player_two = recalculation(one, two)


    x = np.arange(len(comparison_labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, player_one, width, label=player_first)
    rects2 = ax.bar(x + width/2, player_two, width, label=player_second)

    # Add some text for labels, title and custom x-axis tick labels, etc.

    #ax.set_title('Player comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_labels)
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1, 1))

    autolabel(rects1, one, ax)
    autolabel(rects2, two, ax)

    fig.tight_layout()

    return fig

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())

# run the application
if __name__ == "__main__":
    print('main........')  
    app.run(host='0.0.0.0',debug=True,port=8080)