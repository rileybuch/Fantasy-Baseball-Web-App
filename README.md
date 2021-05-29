# Fantasy Baseball Draft Application

## Overview
Don't be overwhelmed by all of the data tables and statistics you can find when you are preparing for your fantasy baseball draft. This platform will give you an easy-to-digest way to do your research. You'll find accurate projections for MLB players' stats, a customizable interface that allows you to input your league's settings and personal risk tolerance in order to generate rankings that are best for you. Easily compare players or see historical performance with graphs and charts that make interpreting the tabular data easier. 

## Installation
To run the app, navigate to the folder team199Final and follow the instructions: 

First, make sure that Docker is installed. Instructions to install Docker can be found at https://docs.docker.com/get-docker/

Once you have Docker, use the terminal/command prompt to navigate into the fantasy_app directory. Once in the directory, run docker-compose up. For example, if the zip file was in the Downloads directory, the following commands would be used to start the application:

cd Downloads/team199final/CODE/fantasy_app  
docker-compose up

The containers for the app and the MySQL database will start building. Once the two containers finish building, the application will automatically start. 

Access the application at localhost:8080. This url will lead to the application's home page, which will present our projections and the visualizations. 

The analysis and models used can be found at https://github.com/rileybuch/fantasy_baseball

## Features

### Projected statistics for players in 2020 

After selecting the batters or pitchers at the home page, there will be a table with projected statistics and ranks for each player in 2020 based on the risk and categories selected. On this page, the following functions are offered:

- Risk tolerance selection: **selection bar** is used for picking low, medium or high risk in ranking players

- Choose stats to rank player: Check the categories in the **boxes** and hit **rank player** button to have the table shown based on the user-defined stats, the default sorting order is determined by projected rank. At least 4 categories are required to rank players.

-  Sorting table: The table could be resorted by any stats by clicking the **header** in the first row of table, for example, click **HR**, then the table will be sorted in ascending or descending order by the amount of HRs.

### Career statistics for individual player

There is also functionality to view historical data for a player. The application gives the ability to:

- View a player's career statistics: After clicking on a **player's name**, a table will display the stats of each season in his career. This table also supports the functionality of sorting by clicking on the **header** of each column.

- Visualization chart on specific category: To easily view trends of a player's performance, each category can be visualized to see the player's performance throughout their career. This is done by selecting a **category** that user would like to view, and clicking the **graph** button.

### Player comparison

Our application offers the ability to compare two players' performance with a chart:

- Compare two players: First, check the boxes in the column of **Compare** for two players that user would like to compare, and hit the **Compare Players** button. Second, choose the season in the **selection bar** and check the categories in the **boxes** that user intends to compare. Finally, click the **Compare These Stats** button. A bar chart with the categories chosen will appear. This offers the user the ability to visually compare player performance rather than comparing raw data.
