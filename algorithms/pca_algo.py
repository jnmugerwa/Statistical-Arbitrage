"""
Based on paper by Avellaneda and Lee.
Long-short based on PCA-derived risk factors.
author: Nathan Mugerwa

To improve: Try different learning models, de-"black box" the PCA process and try
different factor derivation processes (i.e. etfs), **create a more intelligent model,
and more!
----------------------------------------------------------------------------------------
Key Errors:

I. MODEL DOES NOT ACCOUNT FOR TAIL EVENTS (Remember 2008?)
It does not check WHY a stock's idiosyncratic residual is so high. For example, a
stock that announces bankruptcy and drops by 99% in a day would be treated as extremely
cheap and bought up by this algorithm.

II. MODEL MAKES VERY STRONG ASSUMPTIONS
Returns fluctuation is a stationary process, mean-reverting, ... My time as
a discretionary trader makes me confident in eliminating lots of these assumptions
in different iterations of the algorithm. Too theoretical.

III. MODEL PERFORMS BADLY OUT-OF-SAMPLE
Bootstrapping? Hmm... Needs more thought.
"""

import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#######################################################
lookback_days, num_risk_factors = 256, 15


def initialize(context):
    """
       Creates and/or stores all initial data needed for backtesting.

       Parameters
       ----------
       context: An augmented Python dictionary
            From Quantopian: context is an augmented Python dictionary used for maintaining state during your backtest or
            live trading session. context should be used instead of global variables in the algorithm. Properties can be
            accessed using dot notation (context.some_property).

    """
    set_fixed_costs(context)
    algo.schedule_function(
        trade,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=0, minutes=1),
    )

    make_and_attach_pipeline(context)


def set_fixed_costs(context, slippage_bps=0.00, commission_rate=0.00,
                    vol_limit=1.0, min_trade_cost=0.00):
    """
       Sets the fixed costs of the strategy.

       Parameters
       ----------
       context: An augmented Python dictionary
            see above.
        slippage_bps: float
            The slippage, or difference in where the signal was triggered and where the trade was actually executed.
            It is tracked in bps, or basis points.
        commission_rate: float
            Commission paid per-trade.
        vol_limit: float
            Sets a maximum amount of the volume that the strategy can trade.
        min_trade_cost: float
            Minimum cost to enter a trade.

    """
    context.set_slippage(slippage.FixedBasisPointsSlippage(basis_points=slippage_bps, volume_limit=vol_limit))
    context.set_commission(commission.PerShare(cost=commission_rate, min_trade_cost=min_trade_cost))


def make_and_attach_pipeline(context):
    """
       Creates a "pipeline" -- i.e. a screener of stock data. By default, it will gets daily candle-bar data on all US
       equities from Quantopian.

       Parameters
       ----------
       context: An augmented Python dictionary
            See first method's comment.

    """
    base_universe = QTradableStocksUS()
    pipe = Pipeline(
        screen=base_universe,
        columns={
            'open': USEquityPricing.open.latest,
        }
    )
    algo.attach_pipeline(pipe, 'pipeline')


def store_stock_data_from_pipe(context):
    """
       Screens and stores the data from the pipeline initialized above.

       Parameters
       ----------
       context: An augmented Python dictionary
            See first method's comment.W

    """
    context.output = algo.pipeline_output('pipeline')
    context.security_list = context.output.index


def construct_portfolio(context, longs, shorts):
    """
       After deriving factors, training our model, and scoring our stocks we have a set of stocks we'd like to buy or sell.
       This method will construct a portfolio containing those stocks, weighing how much money we put into any one stock
       based on a pre-set cap.

       Parameters
       ----------
       context: An augmented Python dictionary
            See first method's comment.
        longs: list
            The list of stocks we'd like to long/buy.
        shorts: list
            The list of stocks we'd like to short/sell.

    """
    for p in context.portfolio.positions:
        if context.portfolio.positions[p].amount > 0 and p in shorts:
            order_target_percent(p, -0.01)
        elif context.portfolio.positions[p].amount < 0 and p in longs:
            order_target_percent(p, 0.01)
        elif context.portfolio.positions[p].amount > 0 and p in longs:
            pass
        elif context.portfolio.positions[p].amount < 0 and p in shorts:
            pass
        elif context.portfolio.positions[p].amount != 0:
            order_target_percent(p, 0.00)

    for l in longs:
        if context.portfolio.positions[l].amount == 0:
            order_target_percent(l, 0.01)

    for s in shorts:
        if context.portfolio.positions[s].amount == 0:
            order_target_percent(s, -0.01)


def derive_factors_using_pca(price_data):
    """
       Conducts PCA to derive the risk factors for the stocks we screened. Then, it transforms the return data using the PCA
       -derived basis (i.e. pca.fit_transform(returns)).

       Please see "statArbOverview.pdf" in my repository if this doesn't make sense.

       Parameters
       ----------
       context: An augmented Python dictionary
            see above.
        slippage_bps: float
            The slippage, or difference in where the signal was triggered and where the trade was actually executed.
            It is tracked in bps, or basis points.
        commission_rate: float
            Commission paid per-trade.
        vol_limit: float
            Sets a maximum amount of the volume that the strategy can trade.
        min_trade_cost: float
            Minimum cost to enter a trade.

       Returns
       -------
       returns: PANDAS df
            A table giving daily return data for each screened stock.
       pca_returns: PANDAS df
            A table giving PCA-transformed return data for each screened stock.
       stocks: a list of Strings
            The names of the stocks we screened.
    """
    returns = np.log(price_data).diff()[1:]
    returns.dropna(inplace=True, axis=1)  # remove stocks with incomplete histories.

    stocks = returns.columns

    # Creates normalization operator, applies model to data: data has unit variance and mean = 0.
    # Necessary for PCA accuracy.
    returns = StandardScaler().fit_transform(returns)

    # Creates PCA operator, applies to return data. We whiten to ensure independent factors (PCs) downstream.
    pca = PCA(n_components=lookback_days, whiten=True)
    pca_returns = pca.fit_transform(returns)

    return returns, pca_returns, stocks


def train_model(price_data):
    """
       Trains a linear model on the stocks' PCA-transformed returns, returning a table with model prediction and score
       values for each stock.

       Please see "statArbOverview.pdf" in my repository if this doesn't make sense.

       Parameters
       ----------
       price_data: PANDAS dataframe
            A dataframe of price data for our screened stocks.

       Returns
       -------
       pd.DataFrame(predictions): PANDAS dataframe
            A table containing idiosyncratic return prediction and errors ("score") for each stock.
    """
    returns, pca_returns, stocks = derive_factors_using_pca(price_data)

    x_train = pca_returns[:-1, :]
    x_test = pca_returns[-1:, :]

    df = pd.DataFrame(returns, columns=stocks)
    df = df[1:]

    predictions = []

    # Linear regression on the idiosyncratic returns. Goal is to rate deviation of actual idiosyncratic
    # to predicted (what our mean-reverting model expects).
    for stock in stocks:
        y = df[stock]
        m = linear_model.LinearRegression()
        m.fit(x_train, y)
        pred = m.predict(x_test)[0]
        # Scoring: R^2; how well the model predicted the actual returns.
        score = m.score(x_train, y)
        predictions.append({'stock': stock, 'pred': pred, 'score': score})

    return pd.DataFrame(predictions)


def select_stocks_to_trade(prediction_df, num_short=100, num_long=100):
    """
       Once we've scored our stocks, we choose some subset to actually take positions in. This method does that.

       Parameters
       ----------
       prediction_df: PANDAS dataframe
            A table containing idiosyncratic return prediction and errors ("score") for each stock.
       num_short: int
            Number of stocks you'd like to short/sell.
       num_long: int
            Number of stocks you'd like to buy/long.

       Returns
       ----------
        longs: list
            The list of stocks we'd like to long/buy.
        shorts: list
            The list of stocks we'd like to short/sell.

    """
    # Extremely simple selection: If portolio size = k, we short top k/2 and sell bottom k/2.
    prediction_df.sort_values('pred', ascending=False, inplace=True)
    longs = prediction_df[:100]
    shorts = prediction_df[-100:]

    longs.sort_values('score', ascending=False, inplace=True)
    longs = longs['stock'][:num_long].tolist()  # best scores

    shorts.sort_values('score', ascending=False, inplace=True)
    shorts = shorts['stock'][:num_short].tolist()  # best scores

    return longs, shorts


def trade(context, data):
    """
       Executes the trading logic of the algorithm: discerning what to buy and sell, then placing orders.

       Parameters
       ----------
       prediction_df: PANDAS dataframe
            A table containing idiosyncratic return prediction and errors ("score") for each stock.
       data: Quantopian data pipeline
            An equities data pipeline.
    """
    stock_price_table = data.history(context.security_list, fields='price', bar_count=lookback_days, frequency="1d")

    predictions_and_scores = train_model(stock_price_table)

    longs, shorts = select_stocks_to_trade(predictions_and_scores)

    construct_portfolio(context, longs, shorts)
