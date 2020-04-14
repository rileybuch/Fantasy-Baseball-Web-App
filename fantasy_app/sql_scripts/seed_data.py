import csv
import pymysql


insert_query = """INSERT INTO dbo.BattingDemo(Season,Name,Team,Age,G,AB,PA,H,1B,2B,3B,HR,R,RBI,BB,IBB,SO,HBP,SF,SH,GDP,SB,CS,AVG,GB,FB,LD,IFFB,Pitches,Balls,Strikes)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""

batting_file = "batting_data_1996_2019.csv"
data = []

with open(batting_file) as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		data.append((
			row['Season'],
			row['Name'],
			row['Team'],
			row['Age'],
			row['G'],
			row['AB'],
			row['PA'],
			row['H'],
			row['1B'],
			row['2B'],
			row['3B'],
			row['HR'],
			row['R'],
			row['RBI'],
			row['BB'],
			row['IBB'],
			row['SO'],
			row['HBP'],
			row['SF'],
			row['SH'],
			row['GDP'],
			row['SB'],
			row['CS'],
			row['AVG'],
			row['GB'] or 0,
			row['FB'] or 0,
			row['LD'] or 0,
			row['IFFB'] or 0,
			row['Pitches'] or 0,
			row['Balls'] or 0,
			row['Strikes'] or 0,
			))

con = pymysql.connect(
	host='fantasy_baseball_db',
	user='root',
	password='root',
	db='dbo'
	)

curs = con.cursor()
curs.executemany(insert_query, data)
con.commit()
curs.close()
con.close()