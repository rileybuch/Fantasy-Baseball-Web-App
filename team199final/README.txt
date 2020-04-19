Description

Currently, with Major League Baseball (MLB) sports sites, statistics and projections are only given in a tabular format. The data is, usually, only given using standard league settings. This tool is designed to accurately project MLB players' stats and rankings, and give the user the ability to put in their own league settings and get custom player projections and rankings based on those settings. Users have the freedom to select the factors they think are important to customize their own algorithm. This provides transparency to the user and gives them the ability to add or remove certain features that they believe should be part of the algorithm. In addition, one can visualize historic data and compare players using charts. Our platform offers the capability to study multiple players’ statistics with friendly, straightforward charts. With visual tools, we believe that it gives users efficient ways to digest data, make draft and in-season decisions, and find trends. 

Installation

In order to run our platform, Docker needs to be installed.

The instructions to install Docker can be found here:

https://docs.docker.com/get-docker/

Once docker is installed, use the terminal/command prompt to go into the fantasy_app directory. Once in the directory, run docker-compose up. For example, if the zip file was in the Downloads directory, the following commands would be used to start the application:

cd Downloads/team199final/CODE/fantasy_app
docker-compose up

The containers for the app and the MySQL database will start building. Once the two containers finish building, the application will automatically start. 

The application can be accessed at the following url:

localhost:8080

This url will lead to the application's home page. This application will present our projections, but the analysis and models used can be found at:

https://github.com/cse6242teamsport/fantasy_baseball

# Fantasy APP functions
​
## Projected statistics for players in 2020 
​
After selecting the batters or pitchers at the home page, there will be a table with projected statistics and ranks for each player in 2020 based on the risk and categories selected. On this page, the following functions are offered:
​
- Risk tolerance selection: **selection bar** is used for picking low, medium or high risk in ranking players
​
- Choose stats to rank player: Check the categories in the **boxes** and hit **rank player** button to have the table shown based on the user-defined stats, the default sorting order is determined by projected rank. At least 4 categories are required to rank players.

-  Sorting table: The table could be resorted by any stats by clicking the **header** in the first row of table, for example, click **HR**, then the table will be sorted in ascending or descending order by the amount of HRs.
​
## Career statistics for individual player
​
There is also functionality to view historical data for an player. The application gives the ability to:

- View a player's career statistics: After clicking on a **player's name**, a table will display the stats of each season in his career. This table also supports the functionality of sorting by clicking on the **header** of each column.

- Visualization chart on specific category: To easily view trends of a player's performance, each category can be visualized to see the player's performance throughout their career. This is done by selecting a **category** that user would like to view, and clicking the **graph** button.
​
## Player comparison
​
Our application offers the ability to compare two players' performance with a chart:
​
- Compare two players: First, check the boxes in the column of **Compare** for two players that user would like to compare, and hit the **Compare Players** button. Second, choose the season in the **selection bar** and check the categories in the **boxes** that user intends to compare. Finally, click the **Compare These Stats** button. A bar chart with the categories chosen will appear. This offers the user the ability to visually compare player performance rather than comparing raw data.
​
​