import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas
import mplcursors

# get data from excel

def autolabel(rects, player_list):
    """Attach a text label above each bar in *rects*, displaying its height."""
    i = 0
    for rect in rects:
        height = rect.get_height() + 0.5
        ax2.annotate('{}'.format(y2[i]),
                    xy=(rect.get_x() + rect.get_width() / 2, height - 1),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        i+= 1




# --------------SETTINGS--------------------------------
# user select the year & players to compare
player = 'Mike Trout'
G = True
HR = True
AB = False
PA = False
H = False
SB = False
CS = False
AVG = False
R = False
RBI = False
#---------------------------------------------------------



comparison_labels = []
cats = ['G', 'HR', 'AB', 'PA', 'H', 'SB', 'CS', 'AVG', 'R', 'RBI']
cats_boolean = [G, HR, AB, PA, H, SB, CS, AVG, R, RBI]
for i in range(len(cats)):
	if (cats_boolean[i] == True):
		comparison_labels.append(cats[i])

# print(comparison_labels)
# labels = ['HR', 'G', 'AVG', 'SB', 'H', 'R']

col = ["Name", "Season"]
for item in comparison_labels:
	col.append(item)
# print(col)

x = []
y1 = []
y2 = []


reader = csv.reader(open('batting_1996_2020_risk_calc.csv'))
df = pandas.read_csv('batting_1996_2020_risk_calc.csv', usecols = lambda column : column in col)
print(df)

for i in range(len(df)):
	if (df['Name'][i] == player):
		x.append(df['Season'][i])
		y1.append(df[comparison_labels[0]][i])	
		y2.append(df[comparison_labels[1]][i])


fig, ax1 = plt.subplots()

y1max = max(y1)


ax1.set_title(player + "'s " + comparison_labels[1] + " statistics")
ax2 = ax1.twinx()
ax1.plot(x, y1, 'o', color = 'royalblue', pickradius = 5 )
# ax2.plot(x, y2, 'o', color = 'r')

ax1.set_ylim(0, y1max * 1.2)
ax1.set_xlabel('Year')
ax1.set_ylabel(comparison_labels[0], color='b')
ax1.tick_params(axis='y', colors='b')
ax2.set_ylabel(comparison_labels[1], color='r')
ax2.tick_params(axis='y', colors='r')
width = 0.5  # the width of the bars
rects2 = ax2.bar(x, y2, width = width, color = 'tomato')

scale_factor = 0.4
ymin = min(y2)
ymax = max(y2)

ax2.set_ylim(ymin * scale_factor, ymax * (1 + scale_factor))
mplcursors.cursor(hover=True)

autolabel(rects2, y2)

plt.show()



