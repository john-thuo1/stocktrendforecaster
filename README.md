# stocktrendforecaster - Assignment
This Streamlit application & associated code implements a simple stocktrendforecasting tool that allows users to explore and analyze stock market trends, perform trend forecasting using an LSTM model(trained on S&P Market Index) and the stock's closing price, analyze historical volatility of the given stock in a 21 Day window, and visualize simple moving averages(SMA) for a selected stock within 100 and 200 Day Windows. The moving average smoothens out short-term fluctuations and highlights longer-term trends. Traders and analysts often use moving averages in technical analysis to identify trends in financial markets or other time-series data.

[Video Link](https://drive.google.com/file/d/10yVcKtYzL2eAmvdZB_Py9RnJH5byRDwp/view?usp=sharing)  <br /> 

[Streamlit Application Link](https://stocktrendforecastery.streamlit.app/)  <br /> 

[Model Development Collab](https://colab.research.google.com/drive/15SxY-8mcUolcxDaMdQmcssatn0m2Ted0#scrollTo=D-bvST_r5PMu)   <br /> 

# Handling App Errors
To address some potential sources of errors within the application, I have :

- Streamlined user input by enabling direct date selection from the calendar and for Stock Selection Option.
- Limited the date range to mitigate potential errors, providing more control over data inputs and reducing the risk of issues related to incompatible or unexpected date formats.
- Implemented a comprehensive try-except block during model loading to catch any issues associated with the loading process.
- Verified that users can only advance after successfully downloading data from Yahoo Finance during the data retrieval process.
- Cached models and data to improve performance by avoiding redundant computations and loading, reducing the risk of errors related to resource-intensive operations.
- Used conditional plotting and Analysis to ensure that the App does not run into the risk of errors related to missing or invalid data.
   
