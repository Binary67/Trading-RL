import yfinance as yf
import pandas as pd

def DownloadData(Ticker: str, StartDate: str, EndDate: str, Interval: str) -> pd.DataFrame:
    """
    Downloads historical market data from Yahoo Finance.

    Args:
        Ticker (str): The stock ticker symbol (e.g., "AAPL").
        StartDate (str): The start date for the data (YYYY-MM-DD).
        EndDate (str): The end date for the data (YYYY-MM-DD).
        Interval (str): The interval of the data (e.g., "1d", "1h", "1wk").

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the historical market data.
                      The DataFrame will have a single index (Datetime).
                      Returns an empty DataFrame if an error occurs or no data is found.
    """
    try:
        Data = yf.download(Ticker, start=StartDate, end=EndDate, interval=Interval)
        if Data.empty:
            print(f"No data found for {Ticker} from {StartDate} to {EndDate} with interval {Interval}.")
            return pd.DataFrame()
        # Ensure the index is DatetimeIndex and not MultiIndex
        if isinstance(Data.index, pd.MultiIndex):
            # yfinance sometimes returns 'Date' and sometimes 'Datetime' in multi-index scenarios
            if 'Date' in Data.index.names:
                Data.reset_index(inplace=True)
                Data.set_index('Date', inplace=True)
            elif 'Datetime' in Data.index.names:
                Data.reset_index(inplace=True)
                Data.set_index('Datetime', inplace=True)
            else: # Default to the first level if specific names aren't found (less ideal)
                Data.reset_index(level=0, inplace=True)


        # If after reset, index is still not DatetimeIndex or if it's not named, set it
        if not isinstance(Data.index, pd.DatetimeIndex):
             # Attempt to convert the index to Datetime if it's not already
            try:
                Data.index = pd.to_datetime(Data.index)
            except:
                print("Failed to convert index to Datetime. Please check data format.")
                return pd.DataFrame()

        Data.index.name = 'Datetime' # Standardize index name

        # Column Flattening for single ticker download:
        if isinstance(Data.columns, pd.MultiIndex):
            # If yfinance returns columns like ('Price', 'Open') or ('AAPL', 'Open'),
            # we want the second part: 'Open'.
            # This is typically the case if nlevels > 1.
            if Data.columns.nlevels > 1:
                 Data.columns = Data.columns.get_level_values(Data.columns.nlevels - 1)
            # If, after above, columns are still MultiIndex (e.g. some complex structure),
            # this might need more specific handling. For now, this is a common case.

        # yfinance typically returns columns like 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
        # No .title() needed if names are already correct.
        # FeatureEngineer expects these exact names.

        # Addressing unusual yfinance output where all columns are named after the Ticker:
        if not isinstance(Data.columns, pd.MultiIndex) and len(Data.columns) > 0: # Ensure columns are flat and exist
            all_cols_are_ticker = True
            for col_name in Data.columns:
                if col_name.upper() != Ticker.upper(): # Case-insensitive comparison
                    all_cols_are_ticker = False
                    break

            if all_cols_are_ticker:
                print(f"Warning: All column names were '{Data.columns[0]}'. Attempting to rename to OHLCV.")
                if Data.shape[1] == 5: # Common when auto_adjust=True (Open, High, Low, Close, Volume)
                    Data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                elif Data.shape[1] == 6: # Common when auto_adjust=False (Open, High, Low, Close, Adj Close, Volume)
                    Data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                else:
                    print(f"Error: Columns were all '{Data.columns[0]}' but number of columns is {Data.shape[1]}, expected 5 or 6. Cannot reliably rename.")


        return Data
    except Exception as E:
        print(f"An error occurred while downloading data for {Ticker}: {E}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    TickerSymbol = "AAPL"
    StartDateFilter = "2023-01-01"
    EndDateFilter = "2023-12-31"
    IntervalFilter = "1d"

    HistoricalData = DownloadData(TickerSymbol, StartDateFilter, EndDateFilter, IntervalFilter)

    if not HistoricalData.empty:
        print(f"Successfully downloaded data for {TickerSymbol}:")
        print(HistoricalData.head())
        print(f"Index name: {HistoricalData.index.name}")
        print(f"Index type: {type(HistoricalData.index)}")
        print(f"Is index MultiIndex: {isinstance(HistoricalData.index, pd.MultiIndex)}")
    else:
        print(f"Failed to download data for {TickerSymbol}.")

    TickerSymbolSPY = "SPY"
    HistoricalDataSPY = DownloadData(TickerSymbolSPY, StartDateFilter, EndDateFilter, "1wk")
    if not HistoricalDataSPY.empty:
        print(f"\nSuccessfully downloaded data for {TickerSymbolSPY}:")
        print(HistoricalDataSPY.head())
        print(f"Index name: {HistoricalDataSPY.index.name}")
        print(f"Index type: {type(HistoricalDataSPY.index)}")
        print(f"Is index MultiIndex: {isinstance(HistoricalDataSPY.index, pd.MultiIndex)}")
    else:
        print(f"\nFailed to download data for {TickerSymbolSPY}.")
