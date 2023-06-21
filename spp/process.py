import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
import math
from collections.abc import Iterable

from spp import util

# download data from yahoo.finance
def _download_data(yahoo_corp_name="TSLA", output_csv_name="Tesla", download_to_path="", 
                    start="2019-01-01", end="2019-12-31", interval='1d'):
    """
    interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    """
    data = yf.download(yahoo_corp_name, start=start, end=end, interval=interval)
    data.to_csv(download_to_path+output_csv_name+".csv")

# calculate RSI
def _rsi(df, attribute_name, period = 14):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df[attribute_name].copy().diff()
    close_delta[0] = 0  # for the first one

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = period - 1, adjust=True).mean()
    ma_down = down.ewm(com = period - 1, adjust=True).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    rsi = rsi.fillna(100)
    return rsi

class DataPrepare:
    """
    Implements full pipeline of data downloading and preprocessing. Call order is
    init_yahoo, download, prepare
    """
    def __init__(self):
        pass
        
    def init_yahoo(self, tickers, start=None, end=None, 
                 period=None,
                 interval='1d', stages = []):
        """
        Initialize yahoo download properties
        """
        self.tickers = tickers
        self.start = start
        self.end = end
        self.period = period
        self.interval = interval

        self.dataframes = []
        self.stages = stages

    def init_datalist(self, data_list, stages = []):
        """
        Used for predownloaded data. data_list contains all the 
        dowloaded dataframes and stages contains all the prepare
        stages.
        """
        self.dataframes = data_list
        self.stages = stages

    def download(self):
        """
        Download data
        """
        for ticker in self.tickers:
            self.dataframes.append(yf.download(ticker, start=self.start, end=self.end, interval=self.interval, 
                                               period=self.period))

    def add_stage(self, stage):
        """
        Append a single stage
        stage takes and returns a list of DataFrames
        """
        self.stages.append(stage)

    def prepare(self):
        """
        Preprocess and return all the dataframes
        """
        for stage in self.stages:
            self.dataframes = stage(self.dataframes)
        return self.dataframes

def CutStage(attribute_names: list):
    """
    Stage.
    foreach dataframe take out 'attribute_names' columns only
    """
    if type(attribute_names) != list:
        raise util.SPPException(f"CutStage input {type(attribute_names)} not a {list}")
    def _cut_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            dataframe_list[i] = dataframe[attribute_names]
        return dataframe_list
    return _cut_stage

def WindowStage(window_size=15):
    """
    Stage.
    foreach dataframe made data windowed
    """
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
    """
    Stage.
    foreach dataframe append train & test datasets instead
    """
    def _traintest_stage(dataframe_list: list):
        new_dataframe_list = []
        for i, dataframe in enumerate(dataframe_list):
            train_count = round((1. - test) * len(dataframe))
            new_dataframe_list.append(dataframe[:train_count])
            new_dataframe_list.append(dataframe[train_count:])
        return new_dataframe_list
    return _traintest_stage

def TrainValidTestStage(valid=0.2, test=0.2):
    """
    Stage.
    foreach dataframe append train & valid & test datasets instead
    """
    def _trainvalidtest_stage(dataframe_list: list):
        new_dataframe_list = []
        for i, dataframe in enumerate(dataframe_list):
            train_count = round((1. - test - valid) * len(dataframe))
            valid_count = round(valid * len(dataframe))
            new_dataframe_list.append(dataframe[:train_count])
            new_dataframe_list.append(dataframe[train_count:train_count+valid_count])
            new_dataframe_list.append(dataframe[train_count+valid_count:])
        return new_dataframe_list
    return _trainvalidtest_stage

def TargetsSeparateStage(target_names=["Target"]):
    """
    Stage.
    foreach dataframe take out 'target_names' and append it as another dataset
    """
    def _targetsseparate_stage(dataframe_list: list):
        new_dataframe_list = []
        for i, dataframe in enumerate(dataframe_list):
            columns_to_keep = [col for col in dataframe.columns if col not in target_names]
            new_dataframe_list.append(dataframe[columns_to_keep])
            new_dataframe_list.append(dataframe[target_names])
        return new_dataframe_list
    return _targetsseparate_stage

def TargetStage(target, target_name="Target"):
    """
    Stage.
    foreach dataframe find 'target' column and mark it as 'target_name'
    """
    def _target_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            columns = dataframe.columns.to_list()
            if target not in columns:
                raise util.SPPException(f"{target} not in {columns}")
            columns.remove(target)
            dataframe_list[i] =  dataframe_list[i][columns + [target]].rename(columns={target: target_name})
        return dataframe_list
    return _target_stage

def ConcatStage(step: int):
    """
    Stage.
    foreach dataframe concat dataframes with 'step'
    """
    def _concat_stage(dataframe_list: list):
        new_dataframe_list = []
        for start_id in range(min(step, len(dataframe_list))):
            new_dataframe = None
            for current_id in range(start_id, len(dataframe_list), step):
                if new_dataframe is None:
                    new_dataframe = dataframe_list[current_id]
                else:
                    new_dataframe = pd.concat([new_dataframe, dataframe_list[current_id]])
            new_dataframe_list.append(new_dataframe.reset_index(drop=True))
        return new_dataframe_list
    return _concat_stage

def LnProfStage(attribute_name="Close"):
    """
    Stage.
    foreach dataframe compute ln profit from 'attribute_name' column
    [First value is considered 1]
    """
    def _lnprof_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            lnprofs = [1]
            for j in range(1, len(dataframe)):
                price_curr = dataframe[attribute_name].to_list()[j]
                price_prev = dataframe[attribute_name].to_list()[j-1]
                lnprofs.append(math.log(price_curr / price_prev))
            dataframe_list[i][attribute_name+"_LnProf"] = lnprofs
            # dataframe_list[i] = dataframe_list[i][1:]
        return dataframe_list
    return _lnprof_stage

def DivStage(divisible_attr_name, divisor_attr_name, attr_name):
    """
    Stage.
    foreach dataframe create 'attr_name' = 'divisible_attr_name' / 'divisor_attr_name'
    [First value is considered 0]
    """
    def _div_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            dataframe_list[i][attr_name] = dataframe_list[i][divisible_attr_name] / dataframe_list[i][divisor_attr_name]
        return dataframe_list
    return _div_stage

def BinarizeStage(threshold=0., attribute_name="Close_LnProf"):
    """
    Stage.
    foreach dataframe value from 'attribute_name' column apply: 
    1 if value > threshold else 0
    """
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
    """
    Stage.
    foreach dataframe computes real stock cost based on 'prev_price_attribute_name' 
    from logarifmic profit in 'attribute_name'
    ['prev_price_attribute_name' should contain data shifted by one 1 day back]
    """
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
    """
    Stage.
    foreach dataframe drop all the 'attribute_names'
    """
    if type(attribute_names) != list:
        raise util.SPPException(f"DropStage input {type(attribute_names)} not a {list}")
    def _drop_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            columns_to_keep = [col for col in dataframe.columns if col not in attribute_names]
            dataframe_list[i] = dataframe[columns_to_keep]
        return dataframe_list
    return _drop_stage

def PopStage(attribute_names: list, keeper: list):
    """
    Stage.
    foreach dataframe drop all the 'attribute_names' and put them to 'keeper'
    """
    if type(attribute_names) != list:
        raise util.SPPException(f"CutStage input {type(attribute_names)} not a {list}")
    if type(attribute_names) != list:
        raise util.SPPException(f"CutStage input {type(attribute_names)} not a {list}")
    def _pop_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            columns_to_keep = [col for col in dataframe.columns if col not in attribute_names]
            keeper.append(dataframe[attribute_names])
            dataframe_list[i] = dataframe[columns_to_keep]
        return dataframe_list
    return _pop_stage

def RSIColStage(attribute_name="Close", period=14):
    """
    Stage.
    foreach dataframe compute RSI from 'attribute_name'
    """
    def _rsicol_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            dataframe_list[i]["RSI"] = _rsi(dataframe, attribute_name, period)
            # dataframe_list[i] = dataframe_list[i][1:]
        return dataframe_list
    return _rsicol_stage

def TrendSubColStage(attribute_name = "Close"):
    """
    Stage.
    foreach dataframe subs trend from 'attribute_name'
    """
    def _trendsubcol_stage(dataframe_list: list):
        for i, dataframe in enumerate(dataframe_list):
            x = pd.DataFrame(np.arange(len(dataframe)), columns=["Idx"])
            y = dataframe[attribute_name]
            regression = LinearRegression().fit(x, y)
            trend = regression.predict(x)

            dataframe_list[i][attribute_name] = dataframe[attribute_name].to_numpy() - trend
        return dataframe_list
    return _trendsubcol_stage

def TempTrainValidTestStage(valid=0.2, test=0.2):
    """
    Stage
    foreach append train & valid & test datasets instead
    """
    def _trainvalidtest_stage(dataframe_list: list):
        new_dataframe_list = []
        for i, dataframe in enumerate(dataframe_list):
            train_count = round((1. - test - valid) * len(dataframe))
            valid_count = round(valid * len(dataframe))
#             print(train_count, valid_count, len(dataframe))
            new_dataframe_list.append(dataframe[:train_count])
            new_dataframe_list.append(dataframe[train_count:train_count+valid_count])
            new_dataframe_list.append(dataframe[train_count+valid_count:])
        return new_dataframe_list
    return _trainvalidtest_stage

def TempTargetsSeparateStage(target_names=["Target"]):
    """
    Stage
    foreach take out 'target_names' and append it as another dataset
    """
    def _targetsseparate_stage(dataframe_list: list):
        new_dataframe_list = []
        for i, dataframe in enumerate(dataframe_list):
            columns_to_keep = [col for col in dataframe.columns if col not in target_names]
            new_dataframe_list.append(dataframe[columns_to_keep])
            new_dataframe_list.append(dataframe[target_names])
        return new_dataframe_list
    return _targetsseparate_stage

class DefaultPipeline:
    """
    Parent for all the data process pipelines. One should override '_stages(self)->list' method to
    add a new pipeline in the system. 
    """
    def __init__(self, ticker, start=None, end=None, period=None, window_size=15, test_coef=0.2):
        """
        ticker      - stock ticker;
        start, end  - iso strings of date, use this OR period API;
        period      - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max; use this OR start/end API;
        window_size - window size;
        test_coef   - amount of data represent test
        """

        self.concat_need = False
        if isinstance(ticker, str):
            self.tickers = [ticker]
        elif isinstance(ticker, Iterable):
            self.tickers = ticker
            self.concat_need = True
        else:
            raise util.SPPException(f"unknown ticker type {type(ticker)}")

        self.start = start
        self.end = end
        self.period = period
        self.window_size = window_size
        self.test_coef = test_coef
        self.prev_price_keeper = []
        self.true_price_keeper = []

    def __str__(self):
        string = ""
        pref = ""
        for stage in self._stages():
            string += pref + stage.__name__
            pref = ", "
        return f"({string})"

    def __repr__(self):
        return f"{type(self).__name__}{self.__str__()}"

    def _stages(self):
        stages=[]
        return stages
    
    def get_data(self):
        """
        Returns (X_train, y_train, X_test, y_test)
        """
        self.prev_price_keeper = []
        self.true_price_keeper = []

        stages=self._stages()

        # # if multiple tickers were proposed
        # if self.concat_need:
        #     stages.append(ConcatStage(4))

        preprocessor = DataPrepare()
        preprocessor.init_yahoo(self.tickers, start=self.start, end=self.end, period=self.period,
                        stages=stages)
        preprocessor.download()
        to_ret = preprocessor.prepare()
        # # fix buffers if multiple tickers were proposed
        # train_keeper = None
        # test_keeper = None
        # for i, keeper in enumerate(self.prev_price_keeper):
        #     if i % 2 == 0:
        #         train_keeper = pd.concat([train_keeper, keeper])
        #     else:
        #         test_keeper = pd.concat([test_keeper, keeper])
        # self.prev_price_keeper = [train_keeper, test_keeper]

        # train_keeper = None
        # test_keeper = None
        # for i, keeper in enumerate(self.true_price_keeper):
        #     if i % 2 == 0:
        #         train_keeper = pd.concat([train_keeper, keeper])
        #     else:
        #         test_keeper = pd.concat([test_keeper, keeper])
        # self.true_price_keeper = [train_keeper, test_keeper]

        return to_ret

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

    def _get_price_true(self, id):
        return self.true_price_keeper[id].to_numpy()

    def get_train_price(self, y_train_ln_pred):
        """
        Repair ln profit to stock price for train data
        """
        return self._get_price(y_train_ln_pred, 0)

    def get_test_price(self, y_test_ln_pred):
        """
        Repair ln profit to stock price for test data

        Return: pd.DataFrame
                Prediction: Predicted price
                Price_True: Real price
        """
        return self._get_price(y_test_ln_pred, 1)

class BaselineBinPipeline(DefaultPipeline): 
    """
    Basic pipeline that contains close data only
    """
    def _stages(self):
        stages=[
            # Use close price in ln form
            CutStage(["Close"]),
            # RSIColStage("Close"),
            LnProfStage("Close"),
            WindowStage(self.window_size),
            TargetStage(f"Close_LnProf_{self.window_size-1}"),

            # Reduce regression task to classification
            BinarizeStage(0., "Target"),
            DropStage(["Target"]),
            TargetStage("Target_Bin"),

            # Split to train/test and X/y
            # DropStage([f"RSI_{self.window_size-1}"]),
            DropStage([f"Close_{i}" for i in range(self.window_size-2)]),
            TrainTestStage(test=self.test_coef),
            PopStage([f"Close_{self.window_size-2}"], self.prev_price_keeper),
            PopStage([f"Close_{self.window_size-1}"], self.true_price_keeper),
            TargetsSeparateStage(),
        ]
        return stages

class RsiBinPipeline(DefaultPipeline):   
    """
    Basic pipeline extended with RSI
    """     
    def _stages(self):
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
            TrainTestStage(test=self.test_coef),
            PopStage([f"Close_{self.window_size-2}"], self.prev_price_keeper),
            PopStage([f"Close_{self.window_size-1}"], self.true_price_keeper),
            TargetsSeparateStage(),
        ]
        return stages

class HistBinPipeline(DefaultPipeline):        
    def _stages(self):
        stages=[
            # Open processing
            DivStage("Open", "Close", "Open_Rel"),
            DropStage(["Open"]),
            # High processing
            DivStage("High", "Close", "High_Rel"),
            DropStage(["High"]),
            # Low processing
            DivStage("Low", "Close", "Low_Rel"),
            DropStage(["Low"]),
            # AdjClose processing
            DropStage(["Adj Close"]),
            # Volume processing
            DropStage(["Volume"]),
            
            # RSI
            RSIColStage("Close"),
            
            LnProfStage("Close"),
            WindowStage(self.window_size),
            TargetStage(f"Close_LnProf_{self.window_size-1}"),

            BinarizeStage(0., "Target"),
            DropStage(["Target"]),
            TargetStage("Target_Bin"),
            
            DropStage([f"Open_Rel_{self.window_size-1}"]),
            DropStage([f"High_Rel_{self.window_size-1}"]),
            DropStage([f"Low_Rel_{self.window_size-1}"]),

            DropStage([f"Close_{i}" for i in range(self.window_size-2)]),
            DropStage([f"RSI_{self.window_size-1}"]),
            
            TempTrainValidTestStage(valid=self.test_coef,test=self.test_coef),   
            PopStage([f"Close_{self.window_size-2}"], self.prev_price_keeper),
            PopStage([f"Close_{self.window_size-1}"], self.true_price_keeper),
            TempTargetsSeparateStage(),
        ]
        return stages

class HistBinPipeline0Valid(DefaultPipeline):        
    def _stages(self):
        stages=[
            # Open processing
            DivStage("Open", "Close", "Open_Rel"),
            DropStage(["Open"]),
            # High processing
            DivStage("High", "Close", "High_Rel"),
            DropStage(["High"]),
            # Low processing
            DivStage("Low", "Close", "Low_Rel"),
            DropStage(["Low"]),
            # AdjClose processing
            DropStage(["Adj Close"]),
            # Volume processing
            DropStage(["Volume"]),
            
            # RSI
            RSIColStage("Close"),
            
            LnProfStage("Close"),
            WindowStage(self.window_size),
            TargetStage(f"Close_LnProf_{self.window_size-1}"),

            BinarizeStage(0., "Target"),
            DropStage(["Target"]),
            TargetStage("Target_Bin"),
            
            DropStage([f"Open_Rel_{self.window_size-1}"]),
            DropStage([f"High_Rel_{self.window_size-1}"]),
            DropStage([f"Low_Rel_{self.window_size-1}"]),

            DropStage([f"Close_{i}" for i in range(self.window_size-2)]),
            DropStage([f"RSI_{self.window_size-1}"]),
            
            TempTrainValidTestStage(test=self.test_coef,valid=0),   
            PopStage([f"Close_{self.window_size-2}"], self.prev_price_keeper),
            PopStage([f"Close_{self.window_size-1}"], self.true_price_keeper),
            TempTargetsSeparateStage(),
        ]
        return stages
