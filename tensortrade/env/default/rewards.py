from abc import abstractmethod
from tabulate import tabulate
import numpy as np
import pandas as pd

from metric import getDuration
from tensortrade.env.generic import RewardScheme, TradingEnv
from tensortrade.feed.core import Stream, DataFeed
import math


class TensorTradeRewardScheme(RewardScheme):
    """An abstract base class for reward schemes for the default environment.
    """    
    
    def reward(self, env: 'TradingEnv') -> float:
#        self.envFeed = env.observer.feed
        self.renderer_history = pd.DataFrame(env.observer.renderer_history)
        self.broker = env.action_scheme.broker
        #self.informer = env.informer.info(env)
        return self.get_reward(env.action_scheme.portfolio)

    @abstractmethod
    def get_reward(self, portfolio) -> float:
        """Gets the reward associated with current step of the episode.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio associated with the `TensorTradeActionScheme`.

        Returns
        -------
        float
            The reward for the current step of the episode.
        """
        raise NotImplementedError()


class SimpleProfit(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases
    in net worth.

    Parameters
    ----------
    window_size : int
        The size of the look back window for computing the reward.

    Attributes
    ----------
    window_size : int
        The size of the look back window for computing the reward.
    """

    def __init__(self, window_size: int = 1):
        self._window_size = self.default('window_size', window_size)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Rewards the agent for incremental increases in net worth over a
        sliding window.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        float
            The cumulative percentage change in net worth over the previous
            `window_size` time steps.
        """
    
        net_worths = [nw['net_worth'] for nw in portfolio.performance.values()]
        returns = [(b - a) / a for a, b in zip(net_worths[::1], net_worths[1::1])]
        returns = np.array([x + 1 for x in returns[-self._window_size:]]).cumprod() - 1
        return 0 if len(returns) < 1 else returns[-1]


class RiskAdjustedReturns(TensorTradeRewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth,
    while penalizing more volatile strategies.

    Parameters
    ----------
    return_algorithm : {'sharpe', 'sortino'}, Default 'sharpe'.
        The risk-adjusted return metric to use.
    risk_free_rate : float, Default 0.
        The risk free rate of returns to use for calculating metrics.
    target_returns : float, Default 0
        The target returns per period for use in calculating the sortino ratio.
    window_size : int
        The size of the look back window for computing the reward.
    """

    def __init__(self,
                 return_algorithm: str = 'sharpe',
                 risk_free_rate: float = 0.,
                 target_returns: float = 0.,
                 window_size: int = 1) -> None:
        algorithm = self.default('return_algorithm', return_algorithm)

        assert algorithm in ['sharpe', 'sortino']

        if algorithm == 'sharpe':
            return_algorithm = self._sharpe_ratio
        elif algorithm == 'sortino':
            return_algorithm = self._sortino_ratio

        self._return_algorithm = return_algorithm
        self._risk_free_rate = self.default('risk_free_rate', risk_free_rate)
        self._target_returns = self.default('target_returns', target_returns)
        self._window_size = self.default('window_size', window_size)

    def _sharpe_ratio(self, returns: 'pd.Series') -> float:
        """Computes the sharpe ratio for a given series of a returns.

        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.

        Returns
        -------
        float
            The sharpe ratio for the given series of a `returns`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1e-9) / (np.std(returns) + 1e-9)

    def _sortino_ratio(self, returns: 'pd.Series') -> float:
        """Computes the sortino ratio for a given series of a returns.

        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.

        Returns
        -------
        float
            The sortino ratio for the given series of a `returns`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = returns.copy()
        downside_returns[returns < self._target_returns] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        return (expected_return - self._risk_free_rate + 1e-9) / (downside_std + 1e-9)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Computes the reward corresponding to the selected risk-adjusted return metric.

        Parameters
        ----------
        portfolio : `Portfolio`
            The current portfolio being used by the environment.

        Returns
        -------
        float
            The reward corresponding to the selected risk-adjusted return metric.
        """
        net_worths = [nw['net_worth'] for nw in portfolio.performance.values()][-(self._window_size + 1):]
        returns = pd.Series(net_worths).pct_change().dropna()
        risk_adjusted_return = self._return_algorithm(returns)
        return risk_adjusted_return


class PBR(TensorTradeRewardScheme):
    """A reward scheme for position-based returns.

    * Let :math:`p_t` denote the price at time t.
    * Let :math:`x_t` denote the position at time t.
    * Let :math:`R_t` denote the reward at time t.

    Then the reward is defined as,
    :math:`R_{t} = (p_{t} - p_{t-1}) \cdot x_{t}`.

    Parameters
    ----------
    price : `Stream`
        The price stream to use for computing rewards.
    """

    registered_name = "pbr"

    def __init__(self, price: 'Stream') -> None:
        super().__init__()
        self.position = -1

        ## PBR works when commissions are negligible.
        ## Need to modify the following line to make it work for cases where commissions are substantial:
        ###NOTE:  the agent learns to hold its position when abs(r) < commission
        #r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        r = Stream.sensor(price, lambda p: p.value, dtype="float").pct_change()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")
        reward = ((position * r).fillna(0)).rename("reward")
        
        
        self.feed = DataFeed([reward])
        self.feed.compile()
        
        
    def on_action(self, action: int, hasOrder: bool, current_step: int) -> None:
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio') -> float:
        return self.feed.next()["reward"]

    def reset(self) -> None:
        """Resets the `position` and `feed` of the reward scheme."""
        self.position = -1
        self.feed.reset()


class SinglePositionProfit(TensorTradeRewardScheme):
    """A reward scheme for single position return.


    Parameters
    ----------
    price : `Stream`
        The price stream to use for computing rewards.
    """

    registered_name = "spp"

    def __init__(self, price: 'Stream') -> None:
        super().__init__()
        self.position = -1
        self.executed = 0
        r = (Stream.sensor(price, lambda p: p.value, dtype="float").pct_change())#.log10()).diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")
        executed = Stream.sensor(self, lambda rs: rs.executed, dtype="float")
        reward = (position * r - executed).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int, hasOrder: bool, current_step: int) -> None:
        self.position = -1 if action == 0 else 1
        performance = pd.DataFrame.from_dict(portfolio.performance, orient='index')
        net_worths=performance["net_worth"]
        self.executed = len(self.broker.executed)*0.002606

    def get_reward(self, portfolio: 'Portfolio') -> float:
        #n_executed = len(self.broker.executed)/2
        return (self.feed.next()["reward"])# - n_executed*0.002606)

    def reset(self) -> None:
        """Resets the `position` and `feed` of the reward scheme."""
        self.position = -1
        self.feed.reset()
        self.executed = 0


class SimpleProfitBaseInstr(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases
    in net worth.

    Parameters
    ----------
    window_size : int
        The size of the look back window for computing the reward.
    
    Attributes
    ----------
    window_size : int
        The size of the look back window for computing the reward.
    """

    def __init__(self, window_size: int = 1, timeframe='H'):
        self._window_size = self.default('window_size', window_size)
        self.hasOrder = False
        self.current_step = 0
        self.buyTrade_perDay = 0
        self.profitable_trade = 0
        self._reward_metric ={}
        self._reward_metric['total_buyTrades'] = 0
        self._reward_metric['win_ratio'] = 0.0

        
        self.timeframe=timeframe
        self.interval='hours'
        self.unit=1
        self.stepPerDay = 24  
        self.tradeOpenDuration_factor = 1 
        self.maxBuyTrade_perDay_beforePenalty = 2 
        self.minOpenDuration = 4 #unit = self.timeframe


        if self.timeframe == '15T':
            self.stepPerDay = 96
            self.tradeOpenDuration_factor = 1
            self.maxBuyTrade_perDay_beforePenalty = 5
            self.minOpenDuration = 4 
            self.interval='minutes'
            self.unit=15
        elif self.timeframe == '30T':
            self.stepPerDay = 48
            self.tradeOpenDuration_factor = 1
            self.maxBuyTrade_perDay_beforePenalty = 3
            self.minOpenDuration = 2
            self.interval='minutes'
            self.unit=30
        elif self.timeframe == '1H':
            self.stepPerDay = 24
            self.tradeOpenDuration_factor = 1
            self.maxBuyTrade_perDay_beforePenalty = 2
            self.minOpenDuration = 60
            self.interval='hours'
            self.unit=1
        elif self.timeframe == '4H':
            self.stepPerDay = 6
            self.tradeOpenDuration_factor = 1
            self.maxBuyTrade_perDay_beforePenalty = 1
            self.minOpenDuration = 4
            self.interval='hours'
            self.unit=4

        self._reward_metric['duration_sinceLastBuyTrade'] = 0.0
        self._reward_metric['reward_pivot'] = 0.0
        self._reward_metric['reward_profit'] = 0.0
        self._reward_metric['reward_stoRsi'] = 0.0
        self._reward_metric['reward_stoRsiVol'] = 0.0
        self._reward_metric['reward_avg'] = 0.0
        self._reward_metric['reward_duration'] = 0.0
        self._reward_metric['penalty_nTrade'] = 0.0
        self._reward_metric['penalty_stayAtSideLine'] = 0.0
        self._reward_metric['nTrade_perDay'] = 0.0
        
    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Rewards the agent for incremental increases in net worth over a
        sliding window.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        float
            The cumulative percentage change in net worth over the previous
            `window_size` time steps.
        """
        #print(portfolio.performance)
        performance = pd.DataFrame.from_dict(portfolio.performance, orient='index')
        net_worths=performance["net_worth"]#"binance:/USDT:/free"]
        #print(net_worths)
        label = '_norm'
        self._reward_metric['reward_pivot'] = 0.0
        self._reward_metric['reward_profit'] = 0.0
        self._reward_metric['reward_stoRsi'] = 0.0
        self._reward_metric['reward_stoRsiVol'] = 0.0
        self._reward_metric['reward_avg'] = 0.0
        self._reward_metric['reward_duration'] = 0.0
        self._reward_metric['reward_holding'] = 0.0
        self._reward_metric['duration_sinceLastBuyTrade'] = 0.0
        self._reward_metric['penalty_stayAtSideLine'] = 0.0
        penalty_loss = -5.0
        extra_reward_holdNprofit = 2.0
        extra_reward = 3.0
        self._reward_metric['penalty_nTrade'] = 0.0
        self._reward_metric['win_ratio'] = 0.0
        self._profit_ratio = 0.0
        
        scaleFactor = 4.0
        if (self.current_step-1)%(self.stepPerDay) == 0:
            self._reward_metric['nTrade_perDay'] = 0 #self.buyTrade_perDay
            self.buyTrade_perDay = 0
            
        trades=self.broker.trades
        if trades:
            # print('len of trades: ', len(list(trades)))
            if len(list(trades)) > 1:
                previous_trade = trades[list(trades)[-2]][0]
            last_trade = trades[list(trades)[-1]][0]
            last_trade_step = last_trade.step

            # print('trades---', last_trade)
            # print(last_trade.side.value, '---last trade step: ', last_trade_step)
            # print('-------current step: ', self.current_step)
            # print('-------renderer historry\n ', self.renderer_history['close'])
            # print('---total order: ' ,  len(list(trades.values())))

            ### NOTE: self.envFeed.inputs is the next observation in queue, not the current one.
            ###       i.e not the same as current_renderer_history.

            # print('---observer feed:', self.envFeed.inputs[1].name)
            # print('---observer feed values: ', self.envFeed.inputs[1].value)


            ### NOTE: last_renderer_history may or may not be the same as current_renderer_history.
            ###       If current step has order, then both are the same.

            lastTrade_renderer_history = self.renderer_history.iloc[last_trade_step - 1]
            current_renderer_history = self.renderer_history.iloc[self.current_step - 1]
            if len(list(trades)) > 1:
                previousTrade_renderer_history = self.renderer_history.iloc[previous_trade.step - 1]
            
            ##this might be the same when hasOrder is True
            price_range= current_renderer_history['PP_1d'+label] - current_renderer_history['S3_1d'+label] ### TO-BE-Modified
            
            if price_range<=0: print('-------price_range<0: ', price_range)# price_range = 1.0
#            price_range_noNorm = current_renderer_history['PP_1d'] - lastTrade_renderer_history['S3_1d'] ### TO-BE-Modified
            if last_trade.side.value == "buy":
                scaleFactor = 3.0              
                self._reward_metric['duration_sinceLastBuyTrade'] = getDuration(lastTrade_renderer_history['date'], current_renderer_history['date'], self.interval, self.unit)
                if self.hasOrder:
                    self._reward_metric['total_buyTrades'] += 1
                    self.buyTrade_perDay += 1
                    self._reward_metric['nTrade_perDay'] = self.buyTrade_perDay
                    # print(last_trade.price ,' -----BUY ', self._reward_metric['total_buyTrades'] )
                    # print(current_renderer_history[['date', 'open', 'high', 'low', 'close']])

                    self._reward_metric['reward_duration'] = 0.0
                    self._reward_metric['reward_pivot'] = self.reward_at_price(lastTrade_renderer_history, float(lastTrade_renderer_history['close'+label]), True, label)

                    #self._reward_metric['reward_stoRsi'] = 0.0 if lastTrade_renderer_history['stoRsi'+label] >= 0.1 else abs(lastTrade_renderer_history['stoRsi'+label]-0.1)/0.1
                    #self._reward_metric['reward_stoRsiVol'] = 0.0 if lastTrade_renderer_history['stoRsiVol'+label] >= 20.0 else abs(lastTrade_renderer_history['stoRsiVol'+label]-20.0)/20
                    #self._reward_metric['reward_avg'] = 0.0 if lastTrade_renderer_history['avg'+label] >= 30.0 else abs(lastTrade_renderer_history['avg'+label]-30.0)/30.0
                    # self._reward_metric['reward_avg'] =  -1*(lastTrade_renderer_history['avg'+label]-50.0)/1000.0
                    # self._reward_metric['reward_avg'] = self._reward_metric['reward_avg'] + (-1)*(lastTrade_renderer_history['avg_4h'+label]-50.0)/1000.0
                    self._reward_metric['reward_avg'] = -1*(lastTrade_renderer_history['avg'+label] + lastTrade_renderer_history['avg_4h'+label] + lastTrade_renderer_history['avg_1d'+label] - 150.0)/1000
                     
                    # self._reward_metric['reward_stoRsiVol'] =  -1*(lastTrade_renderer_history['stoRsiVol'+label]-50.0)/1000.0
                    # self._reward_metric['reward_stoRsiVol'] = self._reward_metric['reward_stoRsiVol'] + (-1)*(lastTrade_renderer_history['stoRsiVol_4h'+label]-50.0)/1000.0
                    self._reward_metric['reward_stoRsiVol'] = -1*(lastTrade_renderer_history['stoRsiVol'+label] + lastTrade_renderer_history['stoRsiVol_4h'+label] + lastTrade_renderer_history['stoRsiVol_1d'+label] - 150.0)/1000


                    if self.buyTrade_perDay>0 and  self.buyTrade_perDay <= self.maxBuyTrade_perDay_beforePenalty:
                        self._reward_metric['penalty_nTrade'] = -1*(1.3**self.buyTrade_perDay-1.0)+10.0
                    elif self.buyTrade_perDay == 0:
                        self._reward_metric['penalty_nTrade'] = 0.0
                    elif self.buyTrade_perDay > self.maxBuyTrade_perDay_beforePenalty:
                        self._reward_metric['penalty_nTrade'] = self.buyTrade_perDay*2
                    #(self.buyTrade_perDay - self.maxBuyTrade_perDay_beforePenalty)/self.maxBuyTrade_perDay_beforePenalty
        
                    if len(list(trades)) > 1 and previous_trade.side.value == "sell":
                        self._reward_metric['reward_profit'] = -1*(current_renderer_history['close'] - previousTrade_renderer_history['close'])/previousTrade_renderer_history['close']

                    # print('b-------reward profit: ', self._reward_metric['reward_profit'])
                    # print('b-------reward_pivot: ', self._reward_metric['reward_pivot'])
                    # print(' b-------reward_avg: ', self._reward_metric['reward_avg'])
                    # print(' b--------reward_avg: ', self._reward_metric['reward_stoRsiVol'])
                else: ### holding
                    #print('--BUY --holding: ----\n')
                    
                    self._reward_metric['reward_duration']=0.0

                    # if self.buyTrade_perDay == self.maxBuyTrade_perDay_beforePenalty:
                    #     self._reward_metric['penalty_nTrade'] = -0.5
                    # elif self.buyTrade_perDay < self.maxBuyTrade_perDay_beforePenalty:
                    #     self._reward_metric['penalty_nTrade'] = 0.0
                    # elif self.buyTrade_perDay > self.maxBuyTrade_perDay_beforePenalty:
                    #     self._reward_metric['penalty_nTrade'] = self.buyTrade_perDay
                    #(self.buyTrade_perDay - self.maxBuyTrade_perDay_beforePenalty)/self.maxBuyTrade_perDay_beforePenalty
                    self._reward_metric['reward_profit'] = 0.0

                    holding_pnl = (current_renderer_history['close'] - lastTrade_renderer_history['close'])/lastTrade_renderer_history['close']
                    # if holding_profit < -0.15 and holding_profit >= -0.35: # penalize agent when drawdown is more than 25%
                    #     self._reward_metric['reward_profit'] = holding_profit
                    
                    if holding_pnl < -0.08:
                        self._reward_metric['reward_holding'] = holding_pnl 
                    elif holding_pnl > 0.08:
                        self._reward_metric['reward_holding'] = holding_pnl
                    # print('date----', current_renderer_history['date'])
                    # columns =  ['PP_1d', 'R1_1d', 'S1_1d',
                    #   'R2_1d', 'S2_1d', 'R3_1d', 'S3_1d', 'PP_1w', 'R1_1w', 'S1_1w', 'R2_1w',
                    #   'S2_1w', 'R3_1w', 'S3_1w']
                    # tmp = current_renderer_history[[c+label for c in columns]]
                    # #print(tabulate(current_renderer_history.reset_index(), headers='keys', tablefmt='orgtbl'))
                    # print(current_renderer_history[columns], '\n')
                    # print(tmp)
                    # print('\n (current_close - last_buy_price)/price_range ')
                    # print(current_renderer_history['close'], '- ', float(last_trade.price), '/ ', price_range_noNorm, ' = ', (current_renderer_history['close'] - float(last_trade.price))/price_range_noNorm)
                    # print('norm-version---')
                    # print('price_range= current_renderer_history[PP_1d+label] - lastTrade_renderer_history[S3_1d+label]')
                    # print(current_renderer_history['PP_1d'+label], ' - ', lastTrade_renderer_history['S3_1d'+label], ' = ', current_renderer_history['PP_1d'+label] - current_renderer_history['S3_1d'+label]  )
                    # print(current_renderer_history['close'+label], '- ', lastTrade_renderer_history['close'+label], '/ ', price_range, '= ', (current_renderer_history['close'+label] - lastTrade_renderer_history['close'+label])/price_range)
                    # print(current_renderer_history['close'+label], '- ', lastTrade_renderer_history['close'+label], '= ', (current_renderer_history['close'+label] - lastTrade_renderer_history['close'+label]))


                    ###
                    # self._reward_metric['reward_profit'] = (current_renderer_history['close'+label] - lastTrade_renderer_history['close'+label])/price_range

                    # if self._reward_metric['reward_profit'] <= 0: self._reward_metric['reward_profit'] = self._reward_metric['reward_profit'] + penalty_loss
                    # else: self._reward_metric['reward_profit'] = self._reward_metric['reward_profit'] + extra_reward_holdNprofit

                    # if self._reward_metric['duration_sinceLastBuyTrade'] > 2*self.stepPerDay:
                    #     self._reward_metric['reward_duration'] = -0.25*np.log10(self._reward_metric['duration_sinceLastBuyTrade']-2*self.stepPerDay)

                    # if self._reward_metric['duration_sinceLastBuyTrade'] < self.minOpenDuration:
                    #     self._reward_metric['reward_duration'] = 0.0
                    # elif self._reward_metric['duration_sinceLastBuyTrade'] > 7*self.stepPerDay:
                    #     self._reward_metric['reward_duration'] = -0.3*(abs(self._reward_metric['duration_sinceLastBuyTrade']-7*self.stepPerDay))
                    # else:
                    #     self._reward_metric['reward_duration'] = 0.5

                    # if self._reward_metric['reward_profit'] <= 0:
                    #     self._reward_metric['reward_duration'] = self._reward_metric['reward_duration']-0.5

                    # print('h-------holding_pnl: ', holding_pnl)
                    # print('h-------reward profit: ', self._reward_metric['reward_profit'])
                    # print('-------reward_duration: ', self._reward_metric['reward_duration'])
                    #print('b-------reward_pivot: ', self._reward_metric['reward_pivot'])
                    # print('\n')
            elif last_trade.side.value == "sell":
                scaleFactor = 4.0 
                # previousTrade_renderer_history = self.renderer_history.iloc[previous_trade.step - 1]
                self._reward_metric['duration_sinceLastBuyTrade'] = getDuration(previousTrade_renderer_history['date'], lastTrade_renderer_history['date'], self.interval, self.unit)

                #print(previous_trade.step, ' -- self._reward_metric['duration_sinceLastBuyTrade'] -- ', self._reward_metric['duration_sinceLastBuyTrade'])
                if self.hasOrder:
                    # print('-----SELL')
                    #profit_noNorm = float(last_trade.price - previous_trade.price)/price_range_noNorm
                    # print('----profit_noNorm: ', last_trade.price, '-', previous_trade.price, ' ----range: ', price_range_noNorm)
                    # print('previousTrade_renderer_history[close]', previousTrade_renderer_history['close'])
                    # print('---profit_noNorm(%): ', profit_noNorm)
                    #profit = (current_renderer_history['close'+label] - previousTrade_renderer_history['close'+label])#*15.0/price_range
                    profit = (current_renderer_history['close'] - previousTrade_renderer_history['close'])/previousTrade_renderer_history['close']
                    # print('----profit_noNorm: ', last_trade.price, '-', previous_trade.price )
                    # print('----profit: ',  (current_renderer_history['close'+label] - previousTrade_renderer_history['close'+label]), ' ----range: ', price_range)
                    # print('---profit(%): ', profit)
                    
                    # self._reward_metric['reward_pivot'] = self.reward_at_price(lastTrade_renderer_history, float(last_trade.price), False, label)
                    self._reward_metric['reward_pivot'] = self.reward_at_price(lastTrade_renderer_history, float(lastTrade_renderer_history['close'+label]), False, label)

                    self._reward_metric['reward_profit'] = profit
                    if profit > 0: self.profitable_trade+=1
                    # if profit <= 0: self._reward_metric['reward_profit'] = self._reward_metric['reward_profit'] + penalty_loss
                    # else:
                    #     self.profitable_trade+=1
                    #     self._reward_metric['reward_profit'] = self._reward_metric['reward_profit'] + extra_reward

                    self._reward_metric['reward_stoRsi'] = 0.0 if lastTrade_renderer_history['stoRsi'+label] <= 0.9 else abs(lastTrade_renderer_history['stoRsi'+label]-0.9)/0.1
                    #self._reward_metric['reward_stoRsiVol'] = 0.0 if lastTrade_renderer_history['stoRsiVol'+label] <= 80.0 else abs(lastTrade_renderer_history['stoRsiVol'+label]-80)/20
                    #self._reward_metric['reward_avg'] = 0.0 if lastTrade_renderer_history['avg'+label] <= 70.0 else abs(lastTrade_renderer_history['avg'+label]-70)/70
                    # self._reward_metric['reward_avg'] = (lastTrade_renderer_history['avg'+label]-50)/1000
                    # self._reward_metric['reward_avg'] = self._reward_metric['reward_avg'] + (lastTrade_renderer_history['avg_4h'+label]-50)/1000
                    self._reward_metric['reward_avg'] = (lastTrade_renderer_history['avg'+label] + lastTrade_renderer_history['avg_4h'+label] + lastTrade_renderer_history['avg_1d'+label] - 150 )/1000
                    
                    # self._reward_metric['reward_stoRsiVol'] = (lastTrade_renderer_history['stoRsiVol'+label]-50)/1000
                    # self._reward_metric['reward_stoRsiVol'] = self._reward_metric['reward_stoRsiVol'] + (lastTrade_renderer_history['stoRsiVol_4h'+label]-50)/1000 
                    self._reward_metric['reward_stoRsiVol'] = (lastTrade_renderer_history['stoRsiVol'+label] + lastTrade_renderer_history['stoRsiVol_4h'+label] + lastTrade_renderer_history['stoRsiVol_1d'+label] -  150 )/1000
                    
                    if self._reward_metric['duration_sinceLastBuyTrade'] < self.minOpenDuration:
                        self._reward_metric['reward_duration'] = -20.0
                    elif self._reward_metric['duration_sinceLastBuyTrade'] > 14*self.stepPerDay:
                        self._reward_metric['reward_duration'] = -7.5*np.log10((self._reward_metric['duration_sinceLastBuyTrade']))#-14*self.stepPerDay))
                    else:
                        self._reward_metric['reward_duration'] = 20.0

                    if self._reward_metric['reward_profit'] <= 0:
                        self._reward_metric['reward_duration'] = -25.0

                    if self._reward_metric['total_buyTrades'] > 10 and self._reward_metric['total_buyTrades'] < 30:
                        self._profit_ratio = 2.*(net_worths.iloc[-1]-net_worths.iloc[0])/(self._reward_metric['total_buyTrades']*net_worths.iloc[0])
                    elif self._reward_metric['total_buyTrades'] < 10:
                        self._profit_ratio = -0.25#profit
                    elif self._reward_metric['total_buyTrades'] >= 30:
                        self._profit_ratio = 0#.1*(net_worths.iloc[-1]-net_worths.iloc[0])/(self._reward_metric['total_buyTrades']*net_worths.iloc[0])

                    # print('-------profit ratio: (', net_worths.iloc[-1], ' - ', net_worths.iloc[0], ')/(', self._reward_metric['total_buyTrades'], ' * ', net_worths.iloc[0], '= ', self._profit_ratio)
                    # print('-------duration_sinceLastBuyTrade: ', self._reward_metric['duration_sinceLastBuyTrade'])
                    # print('-------reward_duration: ', self._reward_metric['reward_duration'])
                    # print('s-------reward profit: ', self._reward_metric['reward_profit'])
                    # print('s-------reward_pivot: ', self._reward_metric['reward_pivot'])
                    # print('s-------reward_avg: ', self._reward_metric['reward_avg'])
                    #print('s------avg: ', lastTrade_renderer_history['avg'+label], '----', lastTrade_renderer_history['avg'])
                    #print('s-------reward_stoRsiVol: ', self._reward_metric['reward_stoRsiVol'])
                else: ### after sell, stay at sideline 
                    self._reward_metric['reward_pivot'] = 0.0
                    self._reward_metric['reward_duration'] = 0.0
                    duration_sinceLastSellTrade = getDuration(lastTrade_renderer_history['date'], current_renderer_history['date'], self.interval, self.unit)
                    if duration_sinceLastSellTrade > 7*self.stepPerDay:
                        self._reward_metric['penalty_stayAtSideLine'] = 0.7*np.log10(duration_sinceLastSellTrade-7*self.stepPerDay)
                    else:
                        self._reward_metric['penalty_stayAtSideLine'] = 0.0
                        
                    
        if np.isnan(self._reward_metric['reward_pivot']): self._reward_metric['reward_pivot'] = 0

        if self._reward_metric['total_buyTrades'] > 0:
            self._reward_metric['win_ratio'] = self.profitable_trade/self._reward_metric['total_buyTrades']
          
        #total_reward = (2.0*self._reward_metric['reward_profit'] + self._reward_metric['reward_pivot'] + self._reward_metric['reward_duration'] + (self._reward_metric['reward_stoRsi']+self._reward_metric['reward_stoRsiVol']+self._reward_metric['reward_avg'])/3.0)/scaleFactor - self._reward_metric['penalty_nTrade']

        ###31May_PPO_train_myReward_bsh_v15 
        #total_reward = 2.0*self._reward_metric['reward_profit'] + self._reward_metric['reward_pivot'] + self._reward_metric['reward_duration'] + self._reward_metric['reward_stoRsi'] - 1.5*self._reward_metric['penalty_nTrade']

        ###strong ntrade penalyty
        # total_reward = 10.0*self._reward_metric['reward_profit'] + self._reward_metric['reward_pivot'] + self._reward_metric['reward_duration'] + self._reward_metric['reward_stoRsi'] - 5.0*self._reward_metric['penalty_nTrade'] - self._reward_metric['penalty_stayAtSideLine']

        
        #total_reward = 20.0*self._reward_metric['reward_profit'] - 7.5*self._reward_metric['penalty_nTrade'] + 5.0*self._reward_metric['reward_duration'] + 3.0*self._reward_metric['reward_pivot']

        # total_reward = (20.0*self._reward_metric['reward_profit'] + 10.0*self._reward_metric['reward_duration'] + 5.0*self._reward_metric['reward_pivot'])/35.0

        # total_reward = (7.5*self._reward_metric['reward_profit'] - 5.0*self._reward_metric['penalty_nTrade'] + 7.5*self._reward_metric['reward_duration'] + 5.5*self._reward_metric['reward_pivot'] + 2.0*self._profit_ratio)/25.0

        # total_reward =  self._profit_ratio + self._reward_metric['reward_profit'] ### PPO_train_myReward_bsh_v35_profit_ratio/PPO_TradingEnv_0b402_00000_0_2021-08-26_01-15-28/checkpoint_005180/checkpoint-5180
        
        #total_reward =  self._profit_ratio ###PPO_train_myReward_bsh_v35_profit_ratio/PPO_TradingEnv_a1e60_00000_0_2021-09-12_01-13-50/checkpoint_008520/checkpoint-8520

        ## PPO_train_myReward_bsh_v35_profit_ratio/PPO_TradingEnv_2490e_00000_0_2021-09-05_15-16-09
        ## PPO_train_myReward_bsh_v35_profit_ratio/PPO_TradingEnv_bc686_00000_0_2021-09-17_13-33-06
        ## PPO_train_myReward_bsh_v35_profit_ratio/PPO_TradingEnv_0eb08_00000_0_2021-09-18_00-05-20
        # if self._reward_metric['duration_sinceLastBuyTrade'] >= self.minOpenDuration and self._reward_metric['duration_sinceLastBuyTrade']<504:
        #     total_reward = self._reward_metric['reward_profit'] * self._reward_metric['duration_sinceLastBuyTrade']/10.0
        # else:
        #     total_reward = 0.0

        #        total_reward = self._reward_metric['reward_profit']
        total_reward = 20.0*self._reward_metric['reward_profit'] + 10.0*self._reward_metric['reward_pivot'] + 10.0*self._reward_metric['reward_avg'] + 10.0*self._reward_metric['reward_stoRsiVol'] + self._reward_metric['reward_holding']
        
        # print('-----total_buyTrad ', self._reward_metric['total_buyTrades'])
        # print('-----buy_trade_perday', self.buyTrade_perDay)

#        if abs(self._reward_metric['reward_holding'])>0.08:
            # print('-----[reward_profit]: ', self._reward_metric['reward_profit'])    
            # print('-----[reward_pivot]: ' , self._reward_metric['reward_pivot'])
            # # print('-----[reward_stoRsi] :', self._reward_metric['reward_stoRsi'])
            # print('-----[reward_holding]: ', self._reward_metric['reward_holding'])
            # print('-----[reward_stoRsiVol] :', self._reward_metric['reward_stoRsiVol'])
            # print('-----[reward_avg] : ', self._reward_metric['reward_avg'])
            # # print('-----[reward_duration]: ', self._reward_metric['reward_duration'])
            # # print('-----([reward_stoRsi]+reward_stoRsivol+[reward_avg])/3.0 :', (self._reward_metric['reward_stoRsi']+self._reward_metric['reward_stoRsiVol']+self._reward_metric['reward_avg'])/3.0)

            # # print('-----penalty nTrade: -', self._reward_metric['penalty_nTrade'])
            # print('-----holding_pnl/total reward:', self._reward_metric['reward_holding']/total_reward*100)
            # print('-----total reward:', total_reward)  
            # print()
            # print()
        
        return total_reward
            

    def on_action(self, action: int, hasOrder: bool, current_step: int) -> None:
        self.hasOrder = hasOrder
        self.current_step = current_step

    def reward_at_price(self, last_renderer_history: pd.DataFrame, last_trade_price: float, isBuyTrade: bool, label: str ):

        psr_1d = pd.Series(
            [ last_renderer_history["PP_1d"+label], last_renderer_history["S1_1d"+label],
              last_renderer_history["S2_1d"+label], last_renderer_history["S3_1d"+label],
              last_renderer_history["R1_1d"+label], last_renderer_history["R2_1d"+label],
              last_renderer_history["R3_1d"+label], last_trade_price
            ]).sort_values().reset_index(drop=True)
        psr_1w = pd.Series(
            [ last_renderer_history["PP_1w"+label], last_renderer_history["S1_1w"+label],
              last_renderer_history["S2_1w"+label], last_renderer_history["S3_1w"+label],
              last_renderer_history["R1_1w"+label], last_renderer_history["R2_1w"+label],
              last_renderer_history["R3_1w"+label], last_trade_price
            ]).sort_values().reset_index(drop=True)

        # index_1d = pd.Index(psr_1d).get_loc(last_trade_price)
        # index_1w = pd.Index(psr_1w).get_loc(last_trade_price)

        index_1d = psr_1d.tolist().index(last_trade_price)
        index_1w = psr_1w.tolist().index(last_trade_price)

        if isinstance(index_1d, slice):
            print(last_trade_price, ' ---index_1d:\n', index_1d, '\n', psr_1d)
        if isinstance(index_1w, slice):
            print(last_trade_price, ' ---index_1w:\n', index_1w, '\n', psr_1w)
        diff_1d = psr_1d.pct_change()
        diff_1w = psr_1w.pct_change()
        
        pct_diff_dn_1d = 0.0
        pct_diff_up_1d = 0.0
        pct_diff_dn_1w = 0.0
        pct_diff_up_1w = 0.0
        if index_1d > 0 and index_1d < len(psr_1d)-1:
            pct_diff_dn_1d = (last_trade_price - psr_1d[index_1d-1]) / (psr_1d[index_1d+1] - psr_1d[index_1d-1])
            pct_diff_up_1d = (psr_1d[index_1d+1] - last_trade_price) / (psr_1d[index_1d+1] - psr_1d[index_1d-1])
        elif index_1d==0:
            pct_diff_up_1d = (psr_1d[index_1d+1] - last_trade_price) / psr_1d[index_1d+1] 
        elif index_1d==len(psr_1d)-1:
            pct_diff_dn_1d = (last_trade_price - psr_1d[index_1d-1]) / psr_1d[index_1d-1]
            
        if index_1w > 0 and index_1w < len(psr_1w)-1:
            pct_diff_dn_1w = (last_trade_price - psr_1w[index_1w-1]) / (psr_1w[index_1w+1] - psr_1w[index_1w-1])
            pct_diff_up_1w = (psr_1w[index_1w+1] - last_trade_price) / (psr_1w[index_1w+1] - psr_1w[index_1w-1])
        elif index_1w==0:
            pct_diff_up_1w = (psr_1w[index_1w+1] - last_trade_price) / psr_1w[index_1w+1]
        elif index_1w==len(psr_1w)-1:
            pct_diff_dn_1w = (last_trade_price - psr_1w[index_1w-1]) / psr_1w[index_1w-1]

        ###if buy, the entry price should be as close as possible to support.
        ###In that case, both the <pct_diff_up_1d> and <pct_diff_up_1w> (..up.. = distance from resistance)
        ###will be large number (closer to one)
        ###Hence, their product should be 1 in the best case.
        if isBuyTrade:
            reward = pct_diff_up_1d*pct_diff_up_1w  
        else:
            reward = pct_diff_dn_1d*pct_diff_dn_1w

        ##TO-DO negative reward?
        
            
        # print('-----last_trade_prie' , last_trade_price)
        # print(' pivot_support_resistance_1d\n: ', psr_1d)
        # print(' pivot_support_resistance_1w\n: ', psr_1w)

        # print('---- pct_diff_dn_1d ', pct_diff_dn_1d)
        # print('---- pct_diff_up_1d ', pct_diff_up_1d )
        # print('---- pct_diff_dn_1w ', pct_diff_dn_1w )
        # print('---- pct_diff_up_1w ', pct_diff_up_1w)
        # print(' index_1d: ', index_1d)
        # print(' index_1w: ', index_1w)
        # print(' diff_1d: \n', diff_1d)
        # print(' diff_1w: \n', diff_1w)
        # print(' reward: ', reward )
        # print('-------------------------------------------')
        
        # return reward*10.0
        return reward/1000.0
        
    def reset(self) -> None:
        """Resets the `position` and `feed` of the reward scheme."""
        self.hasOrder = False
        self._reward_metric['total_buyTrades'] = 0
        self._reward_metric['win_ratio'] = 0.0
        self._reward_metric['nTrade_perDay'] = 0.0
        self.buyTrade_perDay = 0
        self.profitable_trade = 0
        
_registry = {
    'simple': SimpleProfit,
    'risk-adjusted': RiskAdjustedReturns
}


def get(identifier: str) -> 'TensorTradeRewardScheme':
    """Gets the `RewardScheme` that matches with the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `RewardScheme`

    Returns
    -------
    `TensorTradeRewardScheme`
        The reward scheme associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if identifier is not associated with any `RewardScheme`
    """
    if identifier not in _registry.keys():
        msg = f"Identifier {identifier} is not associated with any `RewardScheme`."
        raise KeyError(msg)
    return _registry[identifier]()
