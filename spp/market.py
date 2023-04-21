import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

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
