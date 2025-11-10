import os
import sys
sys.path.append(os.path.abspath("external/lstm-breakout-predictor"))
import pandas as pd
from simulator.execution_price_simulator import generate_sample_execution_prices, calculate_trade_metrics
from utils.data_loader import DataLoader
from utils.data_writer import DataWriter
import random

class TradeLogger:
    def __init__(self, base_path="."):
        """
        Initialize the TradeLogger with the base path for file operations.
        """
        self.base_path = base_path

    def log_trade(self, trade_json: dict, stockname: str):
        """
        Logs the executed trade to the simulated_trades.xlsx file.

        Args:
            trade_json (dict): JSON object containing trade execution details.
            stockname (str): The stock name or ticker symbol.
        """
        stockname = trade_json.get("symbol", stockname)
        # Paths for input and output files
        data_writer = DataWriter()
        
        data_loader = DataLoader()
        config = data_loader.load_config()
        stockfile_dir = config['DATA']['stockdata_dir']

        excel_file = f"{stockfile_dir}{stockname}.xlsx"
        
        sheet_name = 'price_history'
        
        simulated_trades_path = f"{config['MODEL_PATHS']['model_base_path']}{config['EXECUTION_LOG']['executionlog']}"
        # Read the last row from the stock data file
        stock_data = data_loader.load_data(excel_file,sheet_name)
        price_volume_date = stock_data.sort_values(by='Date', ascending = False).iloc[0][['Open','High','Low','Close','Volume','Date']].values
        
        # Read the smulated execution log
        df_excel = pd.read_excel(simulated_trades_path,sheet_name="execution_log")
        df_excel = df_excel.sort_values(by='ExecutionDate', ascending=True)
        
        # Extract trade details from the trade_json
        trade_signal = trade_json.get("action", "HOLD")
        trade_direction = "LONG" if trade_signal == "BUY" else "SHORT" if trade_signal == "SELL" else "HOLD"
        trade_qty = trade_json.get("quantity", 0.0)
        
        # Check if trade_qty is string and default to 0
        if isinstance(trade_qty, str):
            print(trade_qty, " - is the trade_qty which is string, defaulting to 0")
            trade_qty = 0
        
        if trade_signal == "LONG":
            EntryPrice = min(price_volume_date [0], price_volume_date [1], price_volume_date [3])
            ExitPrice = max(price_volume_date [0], price_volume_date [1], price_volume_date [3])
        else:
            EntryPrice = max(price_volume_date [0], price_volume_date [1], price_volume_date [3])
            ExitPrice = min(price_volume_date [0], price_volume_date [1], price_volume_date [3])
        
        ExecutionTime = f"{random.randint(9,16):0{2}d}:{random.randint(0,59):0{2}d}:{random.randint(0,59):0{2}d}"
        OrderMonth = int(price_volume_date [5].split("-")[1])
        HourOfDay = int(ExecutionTime.split(":")[0])
        
        # Create the new DataFrame
        new_data = {
            'TradeId': int(df_excel.sort_values(by='TradeId', ascending=False)['TradeId'].values[0]) + 1,
            'Ticker': stockname,
            'ExecutionDate': price_volume_date [5],
            'Open': price_volume_date [0],
            'High': price_volume_date [1],
            'Low': price_volume_date [2],
            'Close': price_volume_date [3],
            'EntryPrice': EntryPrice,
            'ExitPrice': ExitPrice,
            'MarketVolume': price_volume_date [4],
            'OrderMonth': OrderMonth,
            'OrderQty': trade_qty,
            'TradeDirection': trade_direction,
            'OrderSubType': 'MARKET',
            'Exchange': 'NSE',
            'Broker': 'ICICI',
            'OrderStatus': 'Fulfilled',
            'ExecutionTime': ExecutionTime,
            'HourOfDay': HourOfDay,
            'ExecutedQty': trade_qty,
            'AvgEntryExecutionPrice': None,
            'AvgExitExecutionPrice': None,
            'TotalEntryTradeValue': None,
            'TotalExitTradeValue': None,
            'EntryBrokerage': None,
            'ExitBrokerage': None,
            'NetEntryAmount': None,
            'NetExitAmount': None,
            'TotalTradeSlippageCost': None,
            'ProfitLoss': None,
            'ClientDematId': '123'
        }
        df_executed_trade = pd.DataFrame(new_data,index=[0])
        df_executed_trade_noNA = df_executed_trade.dropna(axis=1, how='all')
        df_recent_rows = df_excel.tail(10)
        df_execution_log_recent = pd.concat([df_recent_rows, df_executed_trade_noNA], ignore_index=True)
        df_execution_log_recent = df_execution_log_recent.sort_values(by='ExecutionDate', ascending=True)
        vol_list = round(df_execution_log_recent["EntryPrice"].rolling(window=10).std(),2)
        df_execution_log_recent.loc[df_execution_log_recent.index[-1],"volatility"] = vol_list[len(vol_list)-1]
        
        df_last_row = df_execution_log_recent.tail(1)
        df_last_row = df_last_row.reset_index(drop=True)
        avgentry, avgexit = generate_sample_execution_prices(
            df_last_row['EntryPrice'].tolist(),
            df_last_row['ExitPrice'].tolist(),
            df_last_row['MarketVolume'].tolist(),
            df_last_row['volatility'].tolist(),
            df_last_row['HourOfDay'].tolist(),
            df_last_row['OrderMonth'].tolist(),
            df_last_row['OrderQty'].tolist(),
            df_last_row['TradeDirection'].tolist()
        )

        df_last_row['AvgEntryExecutionPrice'] = avgentry
        df_last_row['AvgExitExecutionPrice'] = avgexit

        df_last_row = calculate_trade_metrics(df_last_row)
        
        df_last_row = df_last_row.round(2)
        
        # Write to the simulated_trades.xlsx file
        start_row = len(df_excel) + 1
        with pd.ExcelWriter(simulated_trades_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Write the new DataFrame to the specified sheet starting from the calculated row
            # header=False to avoid writing column headers again if appending to existing data
            # index=False to avoid writing the DataFrame index to Excel
            df_last_row.to_excel(writer, sheet_name="execution_log", startrow=start_row, header=False, index=False)
            
        print(f"Trade logged successfully to {simulated_trades_path}")
