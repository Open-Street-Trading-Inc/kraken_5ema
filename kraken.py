import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
import streamlit as st

# Define functions for buy and sell conditions
def sell_condition(current_candle, previous_candle, row):
    red_candle_signal = previous_candle['close'] < previous_candle['open']
    ema5_below_low = previous_candle['ema5'] < previous_candle['low']
    current_open_within_previous = current_candle['open'] >= previous_candle['low'] and current_candle['open'] <= previous_candle['high']
    close_below_low = row['close'] <= previous_candle['low']



    return (
        red_candle_signal and
        ema5_below_low and
        current_open_within_previous and
        close_below_low
    )

def buy_condition(current_candle, previous_candle, row):
    green_candle_signal = previous_candle['close'] > previous_candle['open']
    ema5_above_high = previous_candle['ema5'] > previous_candle['high']
    current_open_within_previous = current_candle['open'] >= previous_candle['low'] and current_candle['open'] <= previous_candle['high']
    close_above_high = row['close'] >= previous_candle['high']

    return (
        green_candle_signal and
        ema5_above_high and
        current_open_within_previous and
        close_above_high
    )



def get_previous_candle(timestamp):
    df_1 = df.copy()
    previous_candles = df_1[df_1['timestamp'] < timestamp]
    previous_candle = previous_candles.sort_values(by='timestamp')
    previous_candle = previous_candle.iloc[-1]
    return previous_candle.to_dict()

st.title('Kraken Trading Bot')

# Load the data using file uploader
uploaded_file = st.file_uploader("Choose 1 min bar file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)

# Load the data using file uploader
uploaded_file_longer = st.file_uploader("Choose longer period bar file")
if uploaded_file_longer is not None:
    hfdf = pd.read_csv(uploaded_file_longer, header=None)

if st.button('Run'):
    # Define the column names
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']

    # Assign the column names to the dataframe
    df.columns = columns
    hfdf.columns = columns


    # convert the data to float
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    hfdf['open'] = hfdf['open'].astype(float)
    hfdf['high'] = hfdf['high'].astype(float)
    hfdf['low'] = hfdf['low'].astype(float)
    hfdf['close'] = hfdf['close'].astype(float)
    hfdf['volume'] = hfdf['volume'].astype(float)


    # Calculate EMA5 and EMA20
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()




    # Define variables for tracking trades and performance
    in_trade = False
    entry_price = 0
    stop_loss = 0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0


    st.subheader('Trade History')

    hfdf.sort_values(by='timestamp', inplace=True)
    hfdf = hfdf[hfdf['timestamp'] >= df.sort_values(by='timestamp').iloc[2]['timestamp']]

    buy_markers = []
    sell_markers = []
    for index, row in hfdf.iterrows():
        # Get the current row of data


        current_candle = get_previous_candle(row['timestamp'])
        previous_candle = get_previous_candle(current_candle['timestamp'])
        

        buy_signal = buy_condition(current_candle, previous_candle, row)    
        sell_signal = sell_condition(current_candle, previous_candle, row)
        marker = {}
        profit = []

        if buy_signal:
            if not in_trade:
                in_trade = True
                entry_price = row['close']
                stop_loss = row['low']
                st.write('Buy at:', entry_price, "at timestamp:", row['timestamp'])
                marker['timestamp'] = row['timestamp']
                marker['price'] = entry_price
                marker['type'] = 'buy'
                buy_markers.append(marker)
        elif sell_signal:
            if in_trade:
                in_trade = False
                exit_price = row['close']
                st.write('Sell at:', exit_price, "Profit:", exit_price - entry_price, "at timestamp:", row['timestamp'])
                profit.append(exit_price - entry_price)
                total_trades += 1
                if exit_price > entry_price:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                marker['timestamp'] = row['timestamp']
                marker['price'] = exit_price
                marker['type'] = 'sell'
                sell_markers.append(marker)

    st.subheader('Performance Metrics')

    st.write('Total Trades:', total_trades)
    st.write('Winning Trades:', winning_trades)
    st.write('Losing Trades:', losing_trades)
    st.write('Win Rate:', winning_trades / total_trades)
    st.write('Loss Rate:', losing_trades / total_trades)
    st.write('Profit Factor:', winning_trades / losing_trades)
    st.write('Net Profit:', sum(profit))
    st.write('Average Profit:', sum(profit) / total_trades)
    st.write('Average Winning Trade:', sum([p for p in profit if p > 0]) / winning_trades)

    st.write('Average Losing Trade:', sum([p for p in profit if p < 0]) / losing_trades)

    st.subheader('Graphical Visualization')

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price')
    plt.plot(df['timestamp'], df['ema5'], label='EMA5')
    plt.scatter([marker['timestamp'] for marker in buy_markers], [marker['price'] for marker in buy_markers], color='green', label='Buy Signal', marker='^')
    plt.scatter([marker['timestamp'] for marker in sell_markers], [marker['price'] for marker in sell_markers], color='red', label='Sell Signal', marker='v')
            
    plt.legend()
    st.pyplot(plt)

