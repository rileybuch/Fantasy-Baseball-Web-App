from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify, Response, url_for
from flask_mysqldb import MySQL
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json, decimal
import io
import base64
from IPython.display import set_matplotlib_formats
import random
#import mplcursors


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

pitch_stats = ['W', 'L', 'ERA', 'SV', 'IP', 'HR', 'SO', 'WHIP']
bat_stats = ['HR', 'TB', 'R', 'RBI', 'SB', 'AVG', 'OBP', 'SLG']

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
    if request.method == 'POST':
        session['risk'] = request.form.get('risk')
        session['rank_stats'] = request.form.getlist('rank_stats')
    else:
        session['risk'] = 'M'
        session['rank_stats'] = ['HR', 'RBI', 'AVG', 'SB', 'R']
    if len(session['rank_stats']) >= 4:
        return redirect(url_for('show_batter_data'))
    else:
        return ('', 204)

@app.route("/batters-data")
def show_batter_data():    
    return render_template('sorting_bat_modify.html', stats=session['rank_stats'], risk=session['risk'])

@app.route("/choose-batters", methods=['GET', 'POST'])
def choose_batters():
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
            session['seasons'] = seasons
            return redirect(url_for('choose_batter_stats'))
        else:
            return ('', 204)

@app.route("/choose-batters-stats")
def choose_batter_stats():
    return render_template('data.html', players=session['player_list'], seasons=session['seasons'],
        stats=bat_stats)

#----------
@app.route("/image")
def image():
    cur = mysql.connection.cursor()
    # cur.execute("SELECT Name, key_mlbam FROM dbo.Batting")
    cur.execute("SELECT Name, key_mlbam FROM dbo.Batting UNION ALL SELECT Name, key_mlbam FROM dbo.Pitching")
    data = cur.fetchall()
    response = Response(response=json.dumps(data, default=decimal_default), status=200, mimetype="application/json")
    return(response)
#--------

@app.route("/pitchers", methods=['GET', 'POST'])
def pitchers():
    session['player_type'] = 'Pitch' 
    if request.method == 'POST':
        session['risk'] = request.form.get('risk')
        session['rank_stats'] = request.form.getlist('rank_stats')
        session['pitch_position'] = request.form.get('position')
    else:
        session['risk'] = 'M'
        session['rank_stats'] = ['W', 'ERA', 'WHIP', 'SO']
        session['pitch_position'] = 'Starter'
    if len(session['rank_stats']) >= 4:
        return redirect(url_for('show_pitcher_data'))
    else:
        return ('', 204)


@app.route("/pitchers-data")
def show_pitcher_data():    
    return render_template('sorting_pitch_modify.html', stats=session['rank_stats'], risk=session['risk'], position=session['pitch_position'])

@app.route("/choose-pitchers", methods=['GET', 'POST'])
def choose_pitchers():
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
            session['seasons'] = seasons
            return redirect(url_for('choose_pitcher_stats'))
        else:
            return ('', 204)

@app.route("/choose-pitcher-stats")
def choose_pitcher_stats():
    return render_template('data.html', players=session['player_list'], seasons=session['seasons'],
        stats=pitch_stats)

@app.route("/battingdata")  
def get_bat_data():
    stat_len = len(session['rank_stats'])
    if session['risk'] == 'M':
        column_rank = [stat + 'rM' for stat in session['rank_stats']]
        rank_add = ' + '.join(column_rank)
        value_columns = ','.join(bat_stats)
    elif session['risk'] == 'L':
        column_rank = [stat + 'rL' for stat in session['rank_stats']]
        column_value = [stat + 'vL ' + stat for stat in bat_stats]
        rank_add = ' + '.join(column_rank)
        value_columns = ','.join(column_value)
    else:
        column_rank = [stat + 'rH' for stat in session['rank_stats']]
        column_value = [stat + 'vH ' + stat for stat in bat_stats]
        rank_add = ' + '.join(column_rank)
        value_columns = ','.join(column_value)

    average = f'(({rank_add}) / {stat_len})'
    cur = mysql.connection.cursor()
    query = f'SELECT Season, Name, {value_columns}, ROW_NUMBER() OVER (ORDER BY {average}) num, ROW_NUMBER() OVER (ORDER BY {average}) "Rank" FROM dbo.Batting WHERE Season = 2020 ORDER BY {average}'
    cur.execute(query)
    data = cur.fetchall()
    response = Response(response=json.dumps(data, default=decimal_default), status=200, mimetype="application/json")
    return(response)

@app.route("/pitchingdata")
def get_pitch_data():
    stat_len = len(session['rank_stats'])
    if session['risk'] == 'M':
        column_rank = [stat + 'rM' for stat in session['rank_stats']]
        rank_add = ' + '.join(column_rank)
        value_columns = ','.join(pitch_stats)
    elif session['risk'] == 'L':
        column_rank = [stat + 'rL' for stat in session['rank_stats']]
        column_value = [stat + 'vL ' + stat for stat in pitch_stats]
        rank_add = ' + '.join(column_rank)
        value_columns = ','.join(column_value)
    else:
        column_rank = [stat + 'rH' for stat in session['rank_stats']]
        column_value = [stat + 'vH ' + stat for stat in pitch_stats]
        rank_add = ' + '.join(column_rank)
        value_columns = ','.join(column_value)

    average = f'(({rank_add}) / {stat_len})'
    position = session['pitch_position']
    cur = mysql.connection.cursor()
    query = f'SELECT Season, Name, pitcher_type Position, {value_columns}, ROW_NUMBER() OVER (ORDER BY {average}) num, ROW_NUMBER() OVER (ORDER BY {average}) "Rank" FROM dbo.Pitching WHERE Season = 2020 AND pitcher_type = \'{position}\' ORDER BY {average}'
    cur.execute(query)
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
    session['player'] = player_name
    if session['player_type'] == 'Bat':
        cur.execute("SELECT Season, Name, HR, TB, R, RBI, SB, AVG, OBP, SLG, ROW_NUMBER() OVER (ORDER BY Season DESC) num FROM dbo.Batting WHERE Name = %s ORDER BY Season DESC", (player_name,))
    else:
        cur.execute("SELECT Season, Name, pitcher_type, W, L, ERA, SV, IP, HR, SO, WHIP, ROW_NUMBER() OVER (ORDER BY Season) num FROM dbo.Pitching WHERE Name = %s ORDER BY Season DESC", (player_name,))
    data = cur.fetchall()
    response = Response(response=json.dumps(data, default=decimal_default), status=200, mimetype="application/json")
    return(response)

@app.route("/compare", methods=['POST'])
def compare_stats():
    session['chart_stats'] = request.form.getlist('stats')
    session['chart_season'] = request.form.get('season')
    if len(session['chart_stats']) > 0:
        return redirect(url_for('chart'))
    else:
        return ('', 204)

##################################
@app.route("/careerstats", methods=['GET','POST'])
def chart2():
    if request.method == "POST":
        session['stat'] = request.form.get("mode")
        if session['stat']:
            fig2 = make_chart2(session['player'], session['stat'], session['player_type'])
            encoded2 = fig_to_base64_2(fig2)
            encoded2 = encoded2.decode('utf-8')
            return render_template('career_stats.html', image = encoded2)
        else:
            return ('', 204)

# #autolabel chart (same as other chart)
def autolabel(rects, player_list, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    i = 0
    for rect in rects:
        height = rect.get_height() + .5
        ax.annotate('{}'.format(player_list[i]),
                   xy=(rect.get_x() + rect.get_width() / 2, height - 1),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
        i+= 1

# # #make historical progression chart
def make_chart2(player, stat, player_type):
    matplotlib.pyplot.switch_backend('Agg')

    # #stat is an input from the radio button
    comparison_labels = ['G']
    comparison_labels.append(stat)

    col = ["Name", "Season"]
    for item in comparison_labels:
	    col.append(item)
    
    x = []
    y1 = []
    y2 = []

    cur = mysql.connection.cursor()

    if player_type == 'Bat':
        cur.execute("SELECT Season, Name, HR, TB, R, RBI, SB, AVG, OBP, SLG, G FROM dbo.Batting WHERE Name = %s AND Season < 2020", (player,))
    else:
        cur.execute("SELECT Season, Name, W, L, ERA, SV, IP, HR, SO, WHIP, G FROM dbo.Pitching WHERE Name = %s AND Season < 2020", (player,))
    data = cur.fetchall()
    df = pd.DataFrame(data)

    for i in range(len(df)):
	    if (df['Name'][i] == player):
		    x.append(df['Season'][i])
		    y1.append(df[comparison_labels[0]][i])
		    y2.append(df[comparison_labels[1]][i])

    fig, ax1 = plt.subplots()
    if len(y1) == 0:
        return 'This is an empty list'
    else:
        y1max = max(y1)

    ax1.set_title(player + "'s " + comparison_labels[1] + " statistics")
    ax2 = ax1.twinx()
    ax1.plot(x, y1, 'o', color = 'royalblue', pickradius = 5 )
    #ax2.plot(x, y2, 'o', color = 'r')

    ax1.set_ylim(0, y1max * 1.2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel(comparison_labels[0], color='b')
    ax1.tick_params(axis='y', colors='b')
    ax2.set_ylabel(comparison_labels[1], color='r')
    ax2.tick_params(axis='y', colors='r')
    width = 0.5  # the width of the bars
    rects2 = ax2.bar(x, y2, width = width, color = 'tomato')

    scale_factor = decimal.Decimal(0.4)
    ymin = min(y2)
    ymax = max(y2)

    ax2.set_ylim(ymin * scale_factor, ymax * (1 + scale_factor))
    #mplcursors.cursor(hover=True)

    autolabel(rects2, y2, ax2)

    return fig

#what does this part do?
def fig_to_base64_2(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())

##################################

@app.route("/chart")
def chart():
    fig = make_chart(int(session['chart_season']), session['player_list'], session['chart_stats'], session['player_type'])
    encoded = fig_to_base64(fig)
    encoded = encoded.decode('utf-8')
    return render_template('index.html', image=encoded, players = session['player_list'])

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

        if maxval == 0:
            const = 1
        else:
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
        df = pd.read_sql("SELECT Season, Name, HR, TB, R, RBI, SB, AVG, OBP, SLG FROM dbo.Batting WHERE Name IN %s", mysql.connection, params=[tuple(players)])
    else:
        df = pd.read_sql("SELECT Season, Name, W, L, ERA, SV, IP, HR, SO, WHIP FROM dbo.Pitching WHERE Name IN %s", mysql.connection, params=[tuple(players)])

    for i in range(len(df)):
        if (df['Name'][i] == player_first and df['Season'][i] == Year):
            for label in comparison_labels:
                if (label == 'AVG' or label == 'OBP' or label == 'SLG'):
                    one.append(round(df[label][i], 3))
                elif (label == 'ERA' or label == 'WHIP' ):
                    one.append(round(df[label][i], 2))
                elif (label == 'IP'):
                    one.append(round(df[label][i], 1))
                else: one.append(df[label][i])

    #--------------------------------------------------------------------------------------------------------------------
    # set_matplotlib_formats('retina', quality=100)
    #--------------------------------------------------------------------------------------------------------------------

    for i in range(len(df)):
        if (df['Name'][i] == player_second and df['Season'][i] == Year):
            for label in comparison_labels:
                if (label == 'AVG' or label == 'OBP' or label == 'SLG'):
                    two.append(round(df[label][i], 3))
                elif (label == 'ERA' or label == 'WHIP' ):
                    two.append(round(df[label][i], 2))
                elif (label == 'IP' ):
                    two.append(round(df[label][i], 1))
                else: two.append(df[label][i])

    player_one, player_two = recalculation(one, two)


    new_player_one = []
    new_player_two = []

    for i in range(len(player_one)):
        variable = random.randrange(70, 100) / 100
        new_player_one.append(player_one[i] * variable)
        new_player_two.append(player_two[i] * variable)
    x = np.arange(len(comparison_labels))  # the label locations
    width = 0.18  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, new_player_one, width , label= players[0], color = 'orangered' )
    rects2 = ax.bar(x + width/2, new_player_two, width , label= players[1], color = 'dodgerblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_title(str(players[0]) +' vs ' + str(players[1]) , fontweight='bold', fontsize= 12, fontname = 'Arial')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_labels, style = 'italic', fontname = 'Arial', fontsize = 12)
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1, 1), fontsize = 'large', fancybox = True, shadow = True, facecolor = 'lightyellow')
    # x = np.arange(len(comparison_labels))  # the label locations
    # width = 0.2  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width/2, player_one, width, label=player_first)
    # rects2 = ax.bar(x + width/2, player_two, width, label=player_second)

    # # Add some text for labels, title and custom x-axis tick labels, etc.

    # #ax.set_title('Player comparison')
    # ax.set_xticks(x)
    # ax.set_xticklabels(comparison_labels)
    # ax.set_yticks([])
    # ax.legend(bbox_to_anchor=(1, 1))

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