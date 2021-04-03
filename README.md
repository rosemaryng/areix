# Areix Crypto Trading Competition
## Random Forest HFT Strategy
We have adopted the “high-frequency trading” strategy for our trades which we look at 5 min data frames for the desired trading pair. With our algorithm, we frequently detect local minimums for buy and local peaks for sell to generate profit. The metrics we used for the model include Simple Moving Average', 'Bollinger Band', 'StochRSI', and 'MFI'.

Current time frame used is:
start_date = '2021-03-01'
end_date = '2021-04-15'




## RSI Strategy
The adopted strategy is using Stochastic RSI on a 15 min time frame -- when ever K & D values crosses, if it's over 85, we would sell, if it's lower than 25, we buy. This is a mid-risk high-return strategy

