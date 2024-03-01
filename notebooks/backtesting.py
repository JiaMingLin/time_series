import pandas as pd
import numpy as np
import os

class Stock:
    def init(self, date):
        self.date
        self.open
        self.high
        self.low
        self.close

class Strategy:
    def init(self, budget):
        self.ori_budget = budget
        self.accum_budget = budget
        self.repository = 0
    
    def action(self, stock):
        import random
        buy = random.randint(0,1)
        if (buy is True) and (self.repository == 0):
            return 1
        elif (buy is False) and (self.repository > 0):
            return 0

    def return_ratio():
        return
    
def get_week_dates(start_date, end_date, data_dir):

    trading_dates = np.genfromtxt(
        os.path.join(data_dir, 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    date_index_map = dict()
    for index, date in enumerate(trading_dates):
        date_index_map[date] = index
    
    start_index = date_index_map[start_date]
    end_index = date_index_map[end_date]

    return trading_dates[start_index:end_index]
    
def backtesting(stock_index, strategy, budget, start_date, end_date, 
                data_dir = '/home/jiaming/git/time_series/data', dataset='kdd17'):
    
    stock_data = pd.read_csv(os.path.join(data_dir, dataset, 'raw', stock_index+'.csv'))
    week_dates = get_week_dates(start_date, end_date, os.path.join(data_dir, dataset))
    pass