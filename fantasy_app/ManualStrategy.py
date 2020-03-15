import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
from marketsim import *
#from BestPossibleStrategy import *
from indicators import *

def author():
    return 'jjjmbr3'


def testPolicy(symbol = 'JPM', sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):

    df_indicators = get_indicators(sd, ed, symbol)
    df_trades=pd.DataFrame(index=df_indicators.index, columns=[symbol, 'Action'])
    sma = df_indicators['SMA']
    bb_value = df_indicators['bb_value']
    momentum = df_indicators['momentum']
    price = df_indicators['price']
    volatility = df_indicators['volatility']
    holdings=0
    buy_or_sell = 'BUY'
    for i in range(11,df_indicators.shape[0]):
        if (price.iloc[i]< sma.iloc[i]) and (bb_value.iloc[i]<-0.5) and (momentum.iloc[i-1]<momentum.iloc[i]) and volatility.iloc[i] < 2: #(sma.iloc[i]<0.5) and (bb_value.iloc[i]<0) and (momentum.iloc[i]<0):
            if holdings < 0:
                holdings += 2000
                df_trades.iloc[i, 0]=2000
                df_trades.iloc[i, 1]='LONG'
        elif (price.iloc[i]< sma.iloc[i]) and (bb_value.iloc[i]<0) and (momentum.iloc[i-1]<momentum.iloc[i]) and volatility.iloc[i] < 2: #(sma.iloc[i]<0.5) and (bb_value.iloc[i]<0) and (momentum.iloc[i]<0):
            if holdings < 1000:
                df_trades.iloc[i, 1]='LONG'
                holdings += 1000
                df_trades.iloc[i, 0]=1000
                
        elif (price.iloc[i] > sma.iloc[i]>1.0) and (bb_value.iloc[i]>0) and (momentum.iloc[i] > momentum.iloc[i+1]) and volatility.iloc[i] < 2:
            if holdings > 0:
                holdings -= 2000
                df_trades.iloc[i, 0]=-2000
                df_trades.iloc[i, 1]='SHORT'
            elif holdings == 0:
                holdings -= 1000
                df_trades.iloc[i, 0]=-1000
                df_trades.iloc[i, 1]='SHORT'
        
        elif  (i+10<df_indicators.shape[0]) and (momentum.iloc[i] > momentum.iloc[i+1]) and volatility.iloc[i] < 2:
            if holdings > 0:
                holdings -= 1000
                df_trades.iloc[i, 0]=-1000
                df_trades.iloc[i, 1]='SHORT'
                
        else:
            df_trades.iloc[i, 0]=0
        
        
            
    return df_trades


def test_code():
    symbol = 'JPM'
    sv=100000
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    df_indicators = get_indicators(sd, ed, symbol)
    por_trade = testPolicy(symbol, sd, ed, sv)
    long_short = pd.DataFrame(por_trade['Action'].dropna())
    print('-----')
    print(long_short.dropna())
    del por_trade['Action']
    por_trade = por_trade.dropna()
    #por_trade = testPolicy()
    por_trade['Symbol'] = symbol
    por_trade['ORDER'] = por_trade[symbol] > 0
    por_trade['ORDER'].replace(True, 'BUY', inplace=True)
    por_trade['ORDER'].replace(False, 'SELL', inplace=True)
    por_trade['SHARES'] = por_trade[symbol].abs()
    
    del por_trade[symbol] 
#    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#        print(por_trade)
    
    
    port_vals = compute_portvals(por_trade, sv, 9.95, 0.005)
    df_bench=pd.DataFrame(index=port_vals.index, columns=['Symbol','ORDER','SHARES'])
    df_bench['Symbol'] = 'JPM'
    df_bench['ORDER'] = 'BUY'
    df_bench['SHARES'] = 0
    df_bench.loc[port_vals.index.values[0], 'SHARES'] = 1000
   # print(df_bench)
    #return

    bench_vals = compute_portvals(df_bench, sv, 9.95, 0.005)

    port_vals=port_vals/port_vals.values[0]
    bench_vals=bench_vals/bench_vals.values[0]
#    price = df_indicators['price']
    plt.figure(figsize=(20,7))
    plt.gca().set_color_cycle(['red', 'green'])
    port, = plt.plot(port_vals)
    bench, = plt.plot(bench_vals)
#    price_v, = plt.plot(price)
    plt.legend([port, bench], ['Portfolio', 'Benchmark'])
    plt.title("In Sample Portfolio vs Benchmark values for Rule Based Optimal Strategy")
    for i, row in long_short.iterrows():

        if (row['Action']=='SHORT'):
            plt.axvline(x=i, color='black')
        else:
            plt.axvline(x=i, color='blue')
            
    plt.show()
    
    sd=dt.datetime(2010,1,1)
    ed=dt.datetime(2011,12,31)
    df_indicators = get_indicators(sd, ed, symbol)
    por_trade = testPolicy(symbol, sd, ed, sv)
    long_short = pd.DataFrame(por_trade['Action'].dropna())
    print('-----')
    print(long_short.dropna())
    del por_trade['Action']
    por_trade = por_trade.dropna()
    #por_trade = testPolicy()
    por_trade['Symbol'] = symbol
    por_trade['ORDER'] = por_trade[symbol] > 0
    por_trade['ORDER'].replace(True, 'BUY', inplace=True)
    por_trade['ORDER'].replace(False, 'SELL', inplace=True)
    por_trade['SHARES'] = por_trade[symbol].abs()
    
    del por_trade[symbol] 
#    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#        print(por_trade)
    
    
    port_vals = compute_portvals(por_trade, sv, 9.95, 0.005)
    df_bench=pd.DataFrame(index=port_vals.index, columns=['Symbol','ORDER','SHARES'])
    df_bench['Symbol'] = 'JPM'
    df_bench['ORDER'] = 'BUY'
    df_bench['SHARES'] = 0
    df_bench.loc[port_vals.index.values[0], 'SHARES'] = 1000
   # print(df_bench)
    #return

    bench_vals = compute_portvals(df_bench, sv, 9.95, 0.005)
    bench_vals=bench_vals/bench_vals.values[0]
    port_vals=port_vals/port_vals.values[0]
    plt.figure(figsize=(20,7))
    plt.gca().set_color_cycle(['red', 'green'])
    port, = plt.plot(port_vals)
    bench, = plt.plot(bench_vals)
#    price_v, = plt.plot(price)
    plt.legend([port, bench], ['Portfolio', 'Benchmark'])
    plt.title("Out Sample Portfolio vs Benchmark values for Rule Based Optimal Strategy")
    for i, row in long_short.iterrows():

        if (row['Action']=='SHORT'):
            plt.axvline(x=i, color='black')
        else:
            plt.axvline(x=i, color='blue')
            
    plt.show()
    
    


if __name__ == "__main__":
    test_code()
