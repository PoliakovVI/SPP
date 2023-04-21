import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
import math

# download data from yahoo.finance
def _download_data(yahoo_corp_name="TSLA", output_csv_name="Tesla", download_to_path="", 
                    start="2019-01-01", end="2019-12-31", interval='1d'):
    """
    interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    """
    data = yf.download(yahoo_corp_name, start=start, end=end, interval=interval)
    data.to_csv(download_to_path+output_csv_name+".csv")

# calculate RSI
def _rsi(df, periods = 14):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['Close'].copy().diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = periods - 1, adjust=True).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

class DataPrepare:
    def __init__(self):
        pass
        
    def init_yahoo(self, tickers, start="2019-01-01", end="2019-12-31", 
                 period=None,
                 interval='1d', stages = []):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.period = period
        self.interval = interval

        self.dataframes = []
        self.stages = stages

    def init_datalist(self, data_list, stages = []):
        self.dataframes = data_list
        self.stages = stages

    def download(self):
        for ticker in self.tickers:
            self.dataframes.append(yf.download(ticker, start=self.start, end=self.end, interval=self.interval, 
                                               period=self.period))

    def add_stage(self, stage):
        """
        stage takes and returns a list of DataFrames
        """
        self.stages.append(stage)

    def prepare(self):
        for stage in self.stages:
            self.dataframes = stage(self.dataframes)
        return self.dataframes

def CutStage(attribute_names: list):
    if type(attribute_names) != list:
        raise Exception(f"Error: CutStage input {type(attribute_names)} not a {list}")
    def _cut_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            dataframe_list[i] = dataframe[attribute_names]
        return dataframe_list
    return _cut_stage

def WindowStage(window_size=15):
    def _window_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            
            windowed_dataframe_len = len(dataframe) - window_size + 1
            windowed_data = [None for k in range(windowed_dataframe_len)]
            columns = None
            for j in range(windowed_dataframe_len):
                windowed_line = []
                columns = []
                for column in dataframe.columns:
                    windowed_line += dataframe[column][j:j+window_size].to_list()
                    columns += [f"{column}_{k}" for k in range(window_size)]                
                windowed_data[j] = windowed_line
            dataframe_list[i] = pd.DataFrame(windowed_data, columns=columns)
        return dataframe_list
    return _window_stage

def TrainTestStage(test=0.2):
    def _traintest_stage(dataframe_list: list):
        new_dataframe_list = []
        for i, dataframe in enumerate(dataframe_list):
            train_count = round((1. - test) * len(dataframe))
            new_dataframe_list.append(dataframe[:train_count])
            new_dataframe_list.append(dataframe[train_count:])
        return new_dataframe_list
    return _traintest_stage

def TargetsSeparateStage(target_names=["Target"]):
    def _targetsseparate_stage(dataframe_list: list):
        new_dataframe_list = []
        for i, dataframe in enumerate(dataframe_list):
            columns_to_keep = [col for col in dataframe.columns if col not in target_names]
            new_dataframe_list.append(dataframe[columns_to_keep])
            new_dataframe_list.append(dataframe[target_names])
        return new_dataframe_list
    return _targetsseparate_stage

def TargetStage(target, target_name="Target"):
    def _target_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            columns = dataframe.columns.to_list()
            if target not in columns:
                raise Exception(f"Error: {target} not in {columns}")
            columns.remove(target)
            dataframe_list[i] =  dataframe_list[i][columns + [target]].rename(columns={target: target_name})
        return dataframe_list
    return _target_stage

def LnProfStage(attribute_name="Close"):
    """
    reduces dataframe length by 1
    """
    def _lnprof_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            lnprofs = [None]
            for j in range(1, len(dataframe)):
                price_curr = dataframe[attribute_name].to_list()[j]
                price_prev = dataframe[attribute_name].to_list()[j-1]
                lnprofs.append(math.log(price_curr / price_prev))
            dataframe_list[i][attribute_name+"_LnProf"] = lnprofs
            dataframe_list[i] = dataframe_list[i][1:]
        return dataframe_list
    return _lnprof_stage

def BinarizeStage(threshold=0., attribute_name="Close_LnProf"):
    def _binarize_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            binarized = []
            for j in range(0, len(dataframe)):
                curr = dataframe[attribute_name].to_list()[j]
                binarized.append(1 if curr >= threshold else 0)
            dataframe_list[i][attribute_name+"_Bin"] = binarized
        return dataframe_list
    return _binarize_stage

def ExpProfStage(prev_price_attribute_name="Close", attribute_name="Close_LnProf"):
    def _expprof_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            expprofs = []
            for j in range(len(dataframe)):
                ln_price_curr = dataframe[attribute_name].to_list()[j]
                price_prev = dataframe[prev_price_attribute_name].to_list()[j]
                expprofs.append(math.exp(ln_price_curr) * price_prev)
            dataframe_list[i][attribute_name+"_ExpProf"] = expprofs
        return dataframe_list
    return _expprof_stage

def DropStage(attribute_names: list):
    if type(attribute_names) != list:
        raise Exception(f"Error: DropStage input {type(attribute_names)} not a {list}")
    def _drop_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            columns_to_keep = [col for col in dataframe.columns if col not in attribute_names]
            dataframe_list[i] = dataframe[columns_to_keep]
        return dataframe_list
    return _drop_stage

def PopStage(attribute_names: list, keeper: list):
    if type(attribute_names) != list:
        raise Exception(f"Error: CutStage input {type(attribute_names)} not a {list}")
    if type(attribute_names) != list:
        raise Exception(f"Error: CutStage input {type(attribute_names)} not a {list}")
    def _pop_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            columns_to_keep = [col for col in dataframe.columns if col not in attribute_names]
            keeper.append(dataframe[attribute_names])
            dataframe_list[i] = dataframe[columns_to_keep]
        return dataframe_list
    return _pop_stage

def RSIColStage(attribute_name = "Close"):
    def _rsicol_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            dataframe_list[i]["RSI"] = _rsi(dataframe)
            dataframe_list[i] = dataframe_list[i][1:]
        return dataframe_list
    return _rsicol_stage

def TrendSubColStage(attribute_name = "Close"):
    def _trendsubcol_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            x = pd.DataFrame(np.arange(len(dataframe)), columns=["Idx"])
            y = dataframe[attribute_name]
            regression = LinearRegression().fit(x, y)
            trend = regression.predict(x)

            dataframe_list[i][attribute_name] = dataframe[attribute_name].to_numpy() - trend
        return dataframe_list
    return _trendsubcol_stage

class DefaultPipeline:
    """
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    def __init__(self, ticker, period="1y", window_size=15):
        self.tickers = [ticker]
        self.period = period
        self.window_size = window_size
        self.prev_price_keeper = []
        self.true_price_keeper = []

    def _stages(self):
        stages=[]
        return stages
    
    def get_data(self):
        """
        Returns (X_train, y_train, X_test, y_test)
        """
        self.prev_price_keeper = []
        self.true_price_keeper = []

        preprocessor = DataPrepare()
        preprocessor.init_yahoo(self.tickers, start=None, end=None, period=self.period,
                        stages=self._stages())
        preprocessor.download()
        return preprocessor.prepare()

    def _get_price(self, y_ln_pred, train_test_id):
        pred_df = pd.DataFrame(np.array(y_ln_pred), columns=["Prediction"])
        pred_df["Close_Prev"] = self.prev_price_keeper[train_test_id].to_numpy()
        
        preprocessor = DataPrepare()
        preprocessor.init_datalist([pred_df],
                        stages=[
                            ExpProfStage("Close_Prev", "Prediction"),
                        ])
        df = preprocessor.prepare()[0][["Prediction_ExpProf"]]
        df.rename(columns={"Prediction_ExpProf": "Prediction"})
        df["Price_True"] = self.true_price_keeper[train_test_id].to_numpy()
        return df

    def get_train_price(self, y_train_ln_pred):
        """
        ln profit to stock price for train data
        """
        return self._get_price(y_train_ln_pred, 0)

    def get_test_price(self, y_test_ln_pred):
        """
        ln profit to stock price for test data
        """
        return self._get_price(y_test_ln_pred, 1)

class BaselineBinPipeline(DefaultPipeline): 
    def _stages(self):
        stages=[
            CutStage(["Close"]),
            # RSIColStage("Close"),
            LnProfStage("Close"),
            WindowStage(self.window_size),
            TargetStage(f"Close_LnProf_{self.window_size-1}"),

            BinarizeStage(0., "Target"),
            DropStage(["Target"]),
            TargetStage("Target_Bin"),

            # DropStage([f"RSI_{self.window_size-1}"]),
            DropStage([f"Close_{i}" for i in range(self.window_size-2)]),
            TrainTestStage(),
            PopStage([f"Close_{self.window_size-2}"], self.prev_price_keeper),
            PopStage([f"Close_{self.window_size-1}"], self.true_price_keeper),
            TargetsSeparateStage(),
        ]
        return stages

class BinarizedPipeline(DefaultPipeline):        
    def _stages(self):
        """
        Returns (X_train, y_train, X_test, y_test)
        """
        stages=[
            CutStage(["Close"]),
            RSIColStage("Close"),
            LnProfStage("Close"),
            WindowStage(self.window_size),
            TargetStage(f"Close_LnProf_{self.window_size-1}"),

            BinarizeStage(0., "Target"),
            DropStage(["Target"]),
            TargetStage("Target_Bin"),

            DropStage([f"RSI_{self.window_size-1}"]),
            DropStage([f"Close_{i}" for i in range(self.window_size-2)]),
            TrainTestStage(),
            PopStage([f"Close_{self.window_size-2}"], self.prev_price_keeper),
            PopStage([f"Close_{self.window_size-1}"], self.true_price_keeper),
            TargetsSeparateStage(),
        ]
        return stages
