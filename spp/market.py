import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import calendar

from spp import util
from spp import process

class TraderTest:
    def __init__(self, capital, price_predicted, price_true):
        self.capital = capital
        self.stock_count = 0
        self.status_money = True

        self.price_predicted = np.array(price_predicted)
        self.price_true = np.array(price_true)

    def trade(self):
        for i in range(len(self.price_true) - 1):
            current_price = self.price_true[i]
            predcited_price = self.price_predicted[i + 1]

            if self.status_money:
                if predcited_price > current_price:
                    self.stock_count = self.capital / current_price
                    self.capital = 0
                    self.status_money = False
                    # print(f"Buy  {current_price}")
            else:
                if predcited_price < current_price:
                    self.capital = self.stock_count * current_price
                    self.stock_count = 0
                    self.status_money = True
                    # print(f"Sell  {current_price}")

        if not self.status_money:
            self.capital = self.stock_count * self.price_true[-1]
            self.stock_count = 0
            self.status_money = True
        return self.capital

class BinTraderTest:
    def __init__(self, capital, bin_predicted, price_true):
        """
        capital         - start capital for testing
        bin_predicted   - model predictions [0 - demotion, 1 - elevation]
        price_true      - real stock price during the period
        """
        self.capital = capital
        self.stock_count = 0
        self.status_money = True

        self.bin_predicted = np.array(bin_predicted)
        self.price_true = np.array(price_true)

    def trade(self):
        for i in range(len(self.price_true) - 1):
            current_price = self.price_true[i]
            predcited_bin = self.bin_predicted[i + 1]

            if self.status_money:
                if predcited_bin:
                    # if owning money and growth's predicted then buy stocks
                    self.stock_count = self.capital / current_price
                    self.capital = 0
                    self.status_money = False
                    # print(f"Buy  {current_price}")
            else:
                if predcited_bin < current_price:
                    # if owning stock and fall's predicted then sell stocks
                    self.capital = self.stock_count * current_price
                    self.stock_count = 0
                    self.status_money = True
                    # print(f"Sell  {current_price}")

        if not self.status_money:
            self.capital = self.stock_count * self.price_true[-1]
            self.stock_count = 0
            self.status_money = True
        return self.capital

class BinPortfolioTest:
    def __init__(self, tickers, pipeline_class, train_period="1y", test_period="1mo", end_date=None, 
                window_size=15, start_capital=100000):
        """
        tickers         - Itarable of stock tickers
        pipeline_class  - data load pipeline from spp.process
        #_period X[d, mo, y]; X - number of units, [] - unit specificator
                        E.g. 10d - 10 days period
        train_period    - for model training
        test_period     - for model testing
        end_date        - string in iso format - latest date that can be used in the test
        window_size     - window size
        start_capital   - start capital for model traiding
        """

        if end_date is None:   
            end_date = datetime.datetime.now().date().__str__()

        end_datetime = datetime.datetime.fromisoformat(end_date)

        test_years, test_months, test_days = util.parse_period(test_period)
        train_years, train_months, train_days = util.parse_period(train_period)

        train_end_datetime = end_datetime - relativedelta(years=test_years, 
                                                        months=test_months, days=test_days)
        train_start_datetime = train_end_datetime - relativedelta(years=train_years, 
                                                        months=train_months, days=train_days)

        test_start_datetime = train_end_datetime - relativedelta(years=0, 
                                                        months=0, days=window_size)                        

        # train data borders
        self.train_start_date = train_start_datetime.strftime("%Y-%m-%d")
        self.train_end_date = train_end_datetime.strftime("%Y-%m-%d")

        # test data borders
        self.test_start_date = test_start_datetime.strftime("%Y-%m-%d")
        self.end_date = end_date

        self.tickers = tickers
        self.pipeline_class = pipeline_class 
        self.window_size = window_size
        self.start_capital = start_capital

        # load dataset
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepares (X_train, y_train, X_test, y_test) for each ticker
        """
        self.dataset = dict()

        for ticker in self.tickers:
            # pipelines preparation
            train_pipeline = self.pipeline_class(ticker, start=self.train_start_date, 
                                            end=self.train_end_date, 
                                            period=None, window_size=self.window_size, test_coef=0)
            test_pipeline = self.pipeline_class(ticker, start=self.test_start_date, end=self.end_date, 
                                            period=None, window_size=self.window_size, test_coef=0)

            self.dataset[ticker] = dict()

            train_data = train_pipeline.get_data()
            test_data = test_pipeline.get_data()

            self.dataset[ticker]["X_train"] = train_data[0]
            self.dataset[ticker]["y_train"] = train_data[1]
            self.dataset[ticker]["X_test"] = test_data[0]
            self.dataset[ticker]["y_test"] = test_data[1]

            # Real price preparation
            # ! Prediction will not be corret as we 
            # ! proposed bin not ln price
            # ! but it's not used
            self.dataset[ticker]["true_price_train"] = train_pipeline.get_train_price(train_data[1])["Price_True"]
            self.dataset[ticker]["true_price_test"] = test_pipeline.get_train_price(test_data[1])["Price_True"]

    def trade(self, model):
        """
        model - an algorithm class with fit, predict and score methods onboard
        Return: pd.DataFrame
        """

        end_capital_reduction = 0
        max_capital_reduction = 0
        train_acc_reduction = 0
        test_acc_reduction = 0

        data_list = []

        for ticker in self.tickers:
            model.fit(self.dataset[ticker]["X_train"], self.dataset[ticker]["y_train"])
            y_pred = model.predict(self.dataset[ticker]["X_test"])

            bin_traider = BinTraderTest(self.start_capital, y_pred, self.dataset[ticker]["true_price_test"])
            end_capital = bin_traider.trade()

            best_traider = BinTraderTest(self.start_capital, self.dataset[ticker]["y_test"], 
                                        self.dataset[ticker]["true_price_test"])
            max_capital = best_traider.trade()
            
            # metrics compute
            income = end_capital / self.start_capital
            potential = (end_capital - self.start_capital) / (max_capital - self.start_capital + 1e-6)
            train_acc = model.score(self.dataset[ticker]["X_train"], self.dataset[ticker]["y_train"])
            test_acc = model.score(self.dataset[ticker]["X_test"], self.dataset[ticker]["y_test"])

            data_list.append([ticker, train_acc, test_acc, income, potential])

            # reduction
            train_acc_reduction += train_acc
            test_acc_reduction += test_acc
            end_capital_reduction += end_capital
            max_capital_reduction += max_capital

        data_list.insert(0, [
            "AVG", 
            train_acc_reduction / len(self.tickers),
            test_acc_reduction / len(self.tickers),
            end_capital_reduction / (len(self.tickers) * self.start_capital),
            (end_capital_reduction - len(self.tickers) * self.start_capital) / (max_capital_reduction - len(self.tickers) * self.start_capital)
            ]
        )

        return pd.DataFrame(data_list, columns=["Case", "TrainAcc", "TestAcc", "Income", "Potential"])

class BinYearTest:
    def __init__(self, tickers, pipeline_class, train_period="1y", end_date=None, window_size=15):
        """
        tickers         - Itarable of stock tickers
        pipeline_class  - data load pipeline from spp.process
        train_period X[d, mo, y]; X - number of units, [] - unit specificator
                        E.g. 10d - 10 days period
        end_date        - string in iso format - border date to the test
        window_size     - window size
        """

        if type(end_date) == str:  
            pass
        elif end_date is None:
            end_date = datetime.datetime.now().date().__str__()
        else:
            raise util.SPPException(f"unknown end_month type {type(end_date)}")

        # the choosen month should not be in the dataset, so,
        # the begining motnth is the same one year ago
        end_datetime = datetime.datetime.fromisoformat(end_date)
        start_date = (end_datetime - relativedelta(months=13)).strftime("%Y-%m-%d")

        self.start_year = int(start_date.split("-")[0])
        self.start_month = int(start_date.split("-")[1])

        self.tickers = tickers
        self.pipeline_class = pipeline_class 
        self.window_size = window_size
        self.start_capital = 100000

        self.train_period = train_period

        self._prepare_dates()

    def _prepare_dates(self):
        self.dates = []
        start_date = f"{self.start_year}-{self.start_month}-01"

        current_month = self.start_month
        current_year = self.start_year
        
        for month_offset in range(12):
            # update current date
            current_month += 1
            if current_month == 13:
                current_month = 1
                current_year += 1

            # append current date
            end_day = calendar.monthrange(current_year, current_month)[1] 
            end_date = f"{current_year}-{str(current_month).zfill(2)}-{end_day}"
            self.dates.append([calendar.month_abbr[current_month] + "_" + str(current_year), end_date])
            
    def trade(self, model):
        """
        model - an algorithm class with fit, predict and score methods onboard

        Return: pd.DataFrame - year metrics
        """

        year_df = None

        for month_enddate in self.dates:
            bpt = BinPortfolioTest(self.tickers, self.pipeline_class, 
                                        train_period=self.train_period, test_period="1mo", end_date=month_enddate[1], 
                                        window_size=self.window_size, start_capital=self.start_capital)
            # get month results
            month_df = bpt.trade(model)
            month_df = month_df[month_df["Case"] == "AVG"].rename(columns={"Case": "Month"})
            month_df.at[0,'Month'] = month_enddate[0]

            if year_df is None:
                year_df = month_df
            else:
                year_df = pd.concat([year_df, month_df])
        return year_df.reset_index(drop=True)
