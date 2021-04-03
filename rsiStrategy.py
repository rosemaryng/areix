import areix_io as aio
from areix_io.utils import create_report_folder, SideType
import pandas as pd
import numpy as np
import math
# import areix_io.utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


PRED_DAYS = 2 
PCT_CHANGE = 0.004
'''
Data pre processing step
'''
def bollinger_band(data, n_lookback, n_std):
    hlc3 = (data['high'] + data['low'] + data['close']) / 3
    mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std()
    upper = mean + n_std*std
    lower = mean - n_std*std
    return upper, lower
def computeRSI(data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi
# K is blue, D is orange
def stochastic(data, k_window, d_window, window):
    
    # input to function is one column from df
    # containing closing price or whatever value we want to extract K and D from
    
    min_val  = data.rolling(window=window, center=False).min()
    max_val = data.rolling(window=window, center=False).max()
    
    stoch = ( (data - min_val) / (max_val - min_val) ) * 100
    
    K = stoch.rolling(window=k_window, center=False).mean() 
    #K = stoch
    print(K)
    D = K.rolling(window=d_window, center=False).mean() 

    return K, D

# DataFeed
def update_df(df):
    # StochRSI update method
    df['RSI'] = computeRSI(df.close, 6)
    df['K'], df['D'] = stochastic(df['RSI'], 3, 3, 14)
    return df

def get_X(data):
    return data.filter(like='x').values

def get_y(data):
    # use price change in the future 2 days as training label
    # -1 -- change  > -4%
    # 0 -- change is betweeen -4% & +4%
    # 1 -- change is > +4%
    y = data.close.pct_change(PRED_DAYS).shift(-PRED_DAYS)
    y[y.between(-PCT_CHANGE, PCT_CHANGE)] = 0 
    y[y > 0] = 1 # Green
    y[y < 0] = -1 # Red
    return y

def get_clean_Xy(df):
    X = get_X(df)
    y = get_y(df).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]
    return X, y

class RSIStrategy(aio.Strategy):
    num_pre_train = 800

    def initialize(self):
        '''
        Model training step
        '''
        self.info('initialize')
        self.code = 'XRP/USDT'
        self.buy = True

        # unwrap csv data for trading pair
        df = self.ctx.feed[self.code]

        print("df before update with stockrsi")
        print(df)
        
        self.ctx.feed[self.code] = update_df(df)
        self.df = df

        print("updated df")
        print(self.ctx.feed[self.code])

        training_set = df[:self.num_pre_train]
        test_set = df[self.num_pre_train:]

    
    
    def before_trade(self, order):
        return True

    def on_order_ok(self, order):
        self.info(f"{order['side'].name} order {order['id']} ({order['order_type'].name}) executed #{order['quantity']} {order['code']} @ ${order['price']:2f}; Commission: ${order['commission']}; Available Cash: ${self.ctx.available_cash}; Position: #{self.ctx.get_quantity(order['code'])}; Gross P&L : ${order['pnl']}; Net P&L : ${order['pnl_net']}")
        if not order['is_open']:
            self.info(f"Trade closed, pnl: {order['pnl']}========")


    def on_market_start(self):
        # self.info('on_market_start')
        pass

    def on_market_close(self):
        # self.info('on_market_close')
        pass

    def on_order_timeout(self, order):
        self.info(f'on_order_timeout. Order: {order}')
        pass

    def finish(self):
        self.info('finish')

    

    def on_bar(self, tick):
        # K, D = stochastic(self.df, 4, 3, 3) # stochrsi settings
        # print("what is K", K)
        # print("what is D", D)

        k = self.df.at[tick, 'K'] 
        d = self.df.at[tick, 'D']
        curr_price = self.df.at[tick, 'close']
        
        # print(self.context.available_cash)
        if math.isnan(k) or math.isnan(d):
            return
        if k < 25 and d < 25:
            if self.buy and abs(d - k) <= 2:
                print("buy tick", tick, "K: ", k, " D: ", d )
                o1 = self.order_amount(code=self.code,amount=self.context.available_cash,side=SideType.BUY, asset_type='Crypto')
                self.info(f"BUY order {o1['id']} created #{o1['quantity']} ")
                self.buy = not self.buy
        
        if k > 85 and d > 85:
            if (not self.buy) and abs(d - k) <= 2 and curr_price > self.context.trade_records[-1]['price']:    
                print("sell tick", tick, "K: ", k, " D: ", d )
                print(curr_price, ' ', self.context.trade_records[-1]['price'])

                o2 = self.order_amount(code=self.code,amount=(self.context.initial_cash - self.context.available_cash),side=SideType.SELL, price=None, asset_type='Crypto',ioc=True)
                self.info(f"SELL order {o2['id']} created #{o2['quantity']} ")
                self.buy = not self.buy


# Run your strategy:

if __name__ == '__main__':
    aio.set_token('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2NDMyMDUxMTQsImlhdCI6MTYxMjEwMTA1NCwic3ViIjoiZjQzZyAzNWc1dzRoNXc0aCB3NDVoNXc0aHc0NWhqNXdqamh3NTRnIHc0NSBnNXc0ICJ9.CLavu4bIpl64So0F0nYl6g3NfmXqopLfS_UC-9wOgrA') # Only need to run once

    base = create_report_folder()

    start_date = '2021-03-01'
    end_date = '2021-03-12'

    sdf = aio.CryptoDataFeed(
        symbols=['XRP/USDT', 'BTC/USDT'], 
        start_date=start_date, 
        end_date=end_date,  
        interval='15m', 
        order_ascending=True, 
        store_path=base
    )
    feed, idx = sdf.fetch_data()
    benchmark = feed.pop('BTC/USDT')

    mytest = aio.BackTest(
        feed, 
        RSIStrategy, 
        commission_rate=0.001, 
        min_commission=0, 
        trade_at='close', 
        benchmark=benchmark, 
        cash=200000, 
        tradedays=idx, 
        store_path=base
    )

    mytest.start()

    # Strategy
    prefix = ''
    stats = mytest.ctx.statistic.stats(pprint=True, annualization=252, risk_free=0.0442)

    print(stats)

    mytest.contest_output()

