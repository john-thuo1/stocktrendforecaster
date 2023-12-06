import yfinance as yf 
import numpy as np 
import tensorflow as tf 
import pandas as pd 
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as ply


model_path = './standardpoor_model.h5'

@st.cache_resource
def load_model():
    try:
        with tf.device('/cpu:0'): 
            model = tf.keras.models.load_model(model_path, compile=False)
            return model 
    except (FileNotFoundError, RuntimeError) as e:
        st.error(str(e))
        return None
    
# Function to customize the sidebar style
def customize_sidebar():
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background: linear-gradient(45deg, #4E2A84, #964B00);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content .stButton {
            color: #4E2A84;
            background-color: #FFD700;
            border: 2px solid #FFD700;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Customizing the sidebar
customize_sidebar()

# Sidebar introduction
st.sidebar.image('stock_analysis_image.jpg', caption='Stock Analysis', use_column_width=True)
st.sidebar.title("Welcome!")
st.sidebar.write("Explore and analyze stock market trends with this intuitive tool.")


# Main Title

st.title("Stock Market Trend Forecaster and Risk Analyzer")
st.text("""
    Note: The purpose of this tool is to provide an overview of overall trends with respect to the closing price! 
    While it is an important factor to consider when investing, other factors come into play.These factors may 
    include Political Climate, Global Events, Competitive Positioning etc.
""")

start_date = st.date_input("Select Start Date", pd.to_datetime("2010-01-01"), min_value=pd.to_datetime("2009-01-01"))
end_date = st.date_input("Select End Date", pd.to_datetime("2023-12-01"), max_value=pd.to_datetime("2023-12-01"))

@st.cache_resource
def download_data(stock_abbreviation): 
    data = yf.download(stock_abbreviation, start_date, end_date)
    return data

stocks_dict = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOG)": "GOOG",
    "Microsoft (MSFT)": "MSFT",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "UnitedHealth (UNH)": "UNH",
    "JPMorgan Chase (JPM)": "JPM",
}

stock_names = list(stocks_dict.keys())

selected_stock_name = st.selectbox("Select Stock Trend Forecasting from the Stock Options", stock_names)
selected_stocks = stocks_dict.get(selected_stock_name)


if st.button("Download Data"):
    if selected_stocks:
        stock_data = download_data(selected_stocks)
        if stock_data is not None:
            st.subheader('Downloaded Data Description ...')
            st.write(stock_data.describe())
    else:
        st.warning("Please select a stock before downloading data.")
        

if 'stock_data' in locals():  
    st.subheader(f"Trend Forecaster for {selected_stocks} from {start_date} to {end_date}")

    data = stock_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Function to create sequences for prediction
    def create_sequences_for_prediction(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data)-seq_length):
            seq = data[i:i+seq_length]
            target = data[i+seq_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    new_sequence_length = 10
    X_test, y_test = create_sequences_for_prediction(data, new_sequence_length)

    # Reshape the new data for LSTM input (samples, time steps, features)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = load_model()

    predictions = model.predict(X_test)

    # Inverse transform the predictions and actual values to original scale
    predictions_original = scaler.inverse_transform(predictions)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))


    trace_actual = ply.Scatter(x=np.arange(len(y_test_original)), y=y_test_original.flatten(), mode='lines', name='Actual Prices', line=dict(color='blue'))
    trace_predicted = ply.Scatter(x=np.arange(len(y_test_original)), y=predictions_original.flatten(), mode='lines', name='Predicted Prices', line=dict(color='red'))

    # Create layout
    layout = ply.Layout(title='LSTM Predictions vs Actual Prices',
                       xaxis=dict(title='Time (days)'),
                       yaxis=dict(title='Stock Price'),
                       legend=dict(x=0, y=1, traceorder='normal'))

    # Create figure
    fig = ply.Figure(data=[trace_actual, trace_predicted], layout=layout)

    # Show the interactive plot
    st.plotly_chart(fig)

if 'stock_data' in locals():  
    st.subheader(f"Risk Analysis for {selected_stocks} stock")
    st.text("""
        One of the Key Concepts in Risk Analysis for Stock Markets involves Volatility Analysis.
        It is a measure of the dispersion of returns for a given security or market index over time.
        High volatility implies a greater potential for large price swings, indicating increased risk,
        while low volatility suggests more stable and predictable price movements.
    """)

    volatility_window = 21
    
    # Calculate volatility using standard deviation
    stock_data['Volatility'] = stock_data['Close'].pct_change().rolling(window=volatility_window).std()

    # Plot volatility
    fig_volatility = ply.Figure()
    fig_volatility.add_trace(ply.Scatter(x=stock_data.index, y=stock_data['Volatility'], mode='lines', name='Volatility', line=dict(color='orange')))
    fig_volatility.update_layout(title='Historical Volatility Analysis',
                                xaxis=dict(title='Time (days)'),
                                yaxis=dict(title='Volatility'),
                                legend=dict(x=0, y=1, traceorder='normal'))

    st.plotly_chart(fig_volatility)
    
    
if 'stock_data' in locals():  
    st.subheader(f"Moving Average Analysis for {selected_stocks} stock")
    
    # Simple Moving Average for the last 100 & 200 days
    ma_100_days = stock_data['Close'].rolling(window=100).mean()
    ma_200_days = stock_data['Close'].rolling(window=200).mean()
    
    # Create traces for closing prices and 100-day SMA
    trace_close = ply.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Closing Prices', line=dict(color='blue'))
    trace_sma_100 = ply.Scatter(x=stock_data.index, y=ma_100_days, mode='lines', name='100-day SMA', line=dict(color='green'))
    trace_sma_200 = ply.Scatter(x=stock_data.index, y=ma_200_days, mode='lines', name='200-day SMA', line=dict(color='red'))

    # layout
    layout1 = ply.Layout(title='Stock Prices with 100 & 200 - day SMA',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Stock Price'),
                    legend=dict(x=0, y=1, traceorder='normal'))

    # Figure
    fig2 = ply.Figure(data=[trace_close, trace_sma_100, trace_sma_200], layout=layout1)

    st.plotly_chart(fig2)
