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
        returns = np.array([x + 1 for x in returns[-self._window_size:]]).cumprod() -1
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

    def __init__(self, window_size: int = 1):
        self._window_size = self.default('window_size', window_size)
        self.hasOrder = False
        self.current_step = 0
        self.buyTrade_perDay = 0
        self.profitable_trade = 0
        self._reward_metric ={}
        self._reward_metric['total_buyTrades'] = 0
        self._reward_metric['win_ratio'] = 0.0
        
        self._reward_metric['duration_sinceLastBuyTrade'] = 0.0
        self._reward_metric['reward_pivot'] = 0.0
        self._reward_metric['reward_profit'] = 0.0
        self._reward_metric['reward_stoRsi'] = 0.0
        self._reward_metric['reward_stoRsiVol'] = 0.0
        self._reward_metric['reward_avg'] = 0.0
        self._reward_metric['reward_duration'] = 0.0
        self._reward_metric['penalty_nTrade'] = 0.0
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
        # performance = pd.DataFrame.from_dict(portfolio.performance, orient='index')
        # net_worths=performance["net_worth"]#"binance:/USDT:/free"]
        #print(net_worths)
        label = '_norm'
        self._reward_metric['reward_pivot'] = 0.0
        self._reward_metric['reward_profit'] = 0.0
        self._reward_metric['reward_stoRsi'] = 0.0
        self._reward_metric['reward_stoRsiVol'] = 0.0
        self._reward_metric['reward_avg'] = 0.0
        self._reward_metric['reward_duration'] = 0.0
        self._reward_metric['duration_sinceLastBuyTrade'] = 0.0
        penalty_loss = -4.0
        extra_reward_holdNprofit = 2.0
        extra_reward = 4.0
        self._reward_metric['penalty_nTrade'] = 0.0
        self._reward_metric['win_ratio'] = 0.0

        tradeOpenDuration_factor = 1 ##current unit= hours. ##CHANGE denominator if according to price interval (unit)
        stepPerDay = 24  ###CHANGE according to price interval ; also equal to max possible trade per day
        maxBuyTrade_perDay_beforePenalty = 2 ###CHANGE according to price interval
        minOpenDuration = 4 ## current unit is hours, CHANGE according to price interval
        scaleFactor = 4.0
        if (self.current_step-1)%stepPerDay == 0:
            #self._reward_metric['nTrade_perDay'] = self.buyTrade_perDay
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

            
            ##this might be the same when hasOrder is True
            price_range= current_renderer_history['PP_1d'+label] - current_renderer_history['S3_1d'+label] ### TO-BE-Modified
            
            if price_range<=0: print('-------price_range<0: ', price_range)# price_range = 1.0
#            price_range_noNorm = current_renderer_history['PP_1d'] - lastTrade_renderer_history['S3_1d'] ### TO-BE-Modified
            if last_trade.side.value == "buy":
                scaleFactor = 3.0
                # self._reward_metric['duration_sinceLastBuyTrade'] = getDuration(lastTrade_renderer_history['date'], current_renderer_history['date'], 'hours')
                if self.hasOrder:
                    self._reward_metric['total_buyTrades'] += 1
                    self.buyTrade_perDay += 1
                    self._reward_metric['nTrade_perDay'] = self.buyTrade_perDay
                    # print(last_trade.price ,' -----BUY ', self._reward_metric['total_buyTrades'] )
                    # print(current_renderer_history[['date', 'open', 'high', 'low', 'close']])
                    self._reward_metric['reward_pivot'] = self.reward_at_price(lastTrade_renderer_history, float(lastTrade_renderer_history['close'+label]), True, label)

                    self._reward_metric['reward_stoRsi'] = 0.0 if lastTrade_renderer_history['stoRsi_1h'+label] >= 0.1 else abs(lastTrade_renderer_history['stoRsi_1h'+label]-0.1)/0.1
                    self._reward_metric['reward_stoRsiVol'] = 0.0 if lastTrade_renderer_history['stoRsiVol_1h'+label] >= 20.0 else abs(lastTrade_renderer_history['stoRsiVol_1h'+label]-20.0)/20
                    self._reward_metric['reward_avg'] = 0.0 if lastTrade_renderer_history['avg_1h'+label] >= 30.0 else abs(lastTrade_renderer_history['avg_1h'+label]-30.0)/30.0
                    self._reward_metric['reward_duration'] = 0.0

                    self._reward_metric['penalty_nTrade'] = 0.0 if self.buyTrade_perDay <= maxBuyTrade_perDay_beforePenalty else self.buyTrade_perDay#(self.buyTrade_perDay - maxBuyTrade_perDay_beforePenalty)/maxBuyTrade_perDay_beforePenalty
        
                else: ### holding
                    # print('--BUY --holding: ----\n')
                    self._reward_metric['reward_duration']=0.0
                    self._reward_metric['penalty_nTrade'] = 0.0 if self.buyTrade_perDay <= maxBuyTrade_perDay_beforePenalty else self.buyTrade_perDay#(self.buyTrade_perDay - maxBuyTrade_perDay_beforePenalty)/maxBuyTrade_perDay_beforePenalty
                    self._reward_metric['reward_profit'] = 0.0
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
                    self._reward_metric['reward_profit'] = (current_renderer_history['close'+label] - lastTrade_renderer_history['close'+label])/price_range
                    # self._reward_metric['reward_duration'] = 0.0 if self._reward_metric['duration_sinceLastBuyTrade'] < minOpenDuration else self._reward_metric['duration_sinceLastBuyTrade']/tradeOpenDuration_factor
                    # if self._reward_metric['reward_profit'] <= 0: self._reward_metric['reward_duration'] = 0.0
                    if self._reward_metric['reward_profit'] <= 0: self._reward_metric['reward_profit'] = self._reward_metric['reward_profit'] + penalty_loss
                    else: self._reward_metric['reward_profit'] = self._reward_metric['reward_profit'] + extra_reward_holdNprofit

                    
            elif last_trade.side.value == "sell":
                scaleFactor = 4.0 
                previousTrade_renderer_history = self.renderer_history.iloc[previous_trade.step - 1]
                self._reward_metric['duration_sinceLastBuyTrade'] = getDuration(previousTrade_renderer_history['date'], lastTrade_renderer_history['date'], 'hours')
                #print(previous_trade.step, ' -- self._reward_metric['duration_sinceLastBuyTrade'] -- ', self._reward_metric['duration_sinceLastBuyTrade'])
                if self.hasOrder:
                    # print('-----SELL')
                    #profit_noNorm = float(last_trade.price - previous_trade.price)/price_range_noNorm
                    # print('----profit_noNorm: ', last_trade.price, '-', previous_trade.price, ' ----range: ', price_range_noNorm)
                    # print('previousTrade_renderer_history[close]', previousTrade_renderer_history['close'])
                    # print('---profit_noNorm(%): ', profit_noNorm)
                    profit = (current_renderer_history['close'+label] - previousTrade_renderer_history['close'+label])/price_range

                    # print('----profit: ',  (current_renderer_history['close'+label] - previousTrade_renderer_history['close'+label]), ' ----range: ', price_range)
                    # print('---profit(%): ', profit)
                    
                    self.profitable_trade+=1
                    # self._reward_metric['reward_pivot'] = self.reward_at_price(lastTrade_renderer_history, float(last_trade.price), False, label)
                    self._reward_metric['reward_pivot'] = self.reward_at_price(lastTrade_renderer_history, float(lastTrade_renderer_history['close'+label]), False, label)

                    self._reward_metric['reward_profit'] = profit 
                    if profit <= 0: self._reward_metric['reward_profit'] = self._reward_metric['reward_profit'] + penalty_loss
                    else: self._reward_metric['reward_profit'] = self._reward_metric['reward_profit'] + extra_reward

                    self._reward_metric['reward_stoRsi'] = 0.0 if lastTrade_renderer_history['stoRsi_1h'+label] <= 0.9 else abs(lastTrade_renderer_history['stoRsi_1h'+label]-0.9)/0.1
                    self._reward_metric['reward_stoRsiVol'] = 0.0 if lastTrade_renderer_history['stoRsiVol_1h'+label] <= 80.0 else abs(lastTrade_renderer_history['stoRsiVol_1h'+label]-80)/20
                    self._reward_metric['reward_avg'] = 0.0 if lastTrade_renderer_history['avg_1h'+label] <= 70.0 else abs(lastTrade_renderer_history['avg_1h'+label]-70)/30
                    self._reward_metric['reward_duration'] = 0.0 if self._reward_metric['duration_sinceLastBuyTrade'] < minOpenDuration else self._reward_metric['duration_sinceLastBuyTrade']/tradeOpenDuration_factor
                    if self._reward_metric['reward_profit'] <= 0: self._reward_metric['reward_duration'] = 0.0
                    
                else: ### after sell, stay at sideline 
                    self._reward_metric['reward_pivot'] = 0.0
                    self._reward_metric['reward_duration'] = 0.0

        if np.isnan(self._reward_metric['reward_pivot']): self._reward_metric['reward_pivot'] = 0

        if self._reward_metric['total_buyTrades'] > 0:
            self._reward_metric['win_ratio'] = self.profitable_trade/self._reward_metric['total_buyTrades']
        
        
        #total_reward = (2.0*self._reward_metric['reward_profit'] + self._reward_metric['reward_pivot'] + self._reward_metric['reward_duration'] + (self._reward_metric['reward_stoRsi']+self._reward_metric['reward_stoRsiVol']+self._reward_metric['reward_avg'])/3.0)/scaleFactor - self._reward_metric['penalty_nTrade']

        ###31May_PPO_train_myReward_bsh_v15 
        #total_reward = 2.0*self._reward_metric['reward_profit'] + self._reward_metric['reward_pivot'] + self._reward_metric['reward_duration'] + self._reward_metric['reward_stoRsi'] - 1.5*self._reward_metric['penalty_nTrade']

        ###strong ntrade penalyty
        total_reward = 2.0*self._reward_metric['reward_profit'] + self._reward_metric['reward_pivot'] + self._reward_metric['reward_duration'] + self._reward_metric['reward_stoRsi'] - 8*self._reward_metric['penalty_nTrade']

        
        #total_reward = 2.0*self._reward_metric['reward_pivot'] + self._reward_metric['reward_duration'] - 1.5*self._reward_metric['penalty_nTrade']

        # print('-----total_buyTrad ', self._reward_metric['total_buyTrades'])
        # print('-----buy_trade_perday', self.buyTrade_perDay)

        
        #print('-----self._reward_metric[reward_profit]: ', self._reward_metric['reward_profit'])    
        # print('-----self._reward_metric['reward_pivot']: ' , self._reward_metric['reward_pivot'])
        # print('-----self._reward_metric['reward_stoRsi'] :', self._reward_metric['reward_stoRsi'])
        # print('-----self._reward_metric['reward_stoRsiVol'] :', self._reward_metric['reward_stoRsiVol'])
        # print('-----self._reward_metric['reward_avg'] : ', self._reward_metric['reward_avg'])
        # print('-----self._reward_metric['reward_duration']: ', self._reward_metric['reward_duration'])
        # print('-----(self._reward_metric['reward_stoRsi']+reward_stoRsi+self._reward_metric['reward_avg'])/3.0 :', (self._reward_metric['reward_stoRsi']+reward_stoRsi+self._reward_metric['reward_avg'])/3.0)

        # print('-----penalty nTrade: -', self._reward_metric['penalty_nTrade'])
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

        index_1d = pd.Index(psr_1d).get_loc(last_trade_price)
        index_1w = pd.Index(psr_1w).get_loc(last_trade_price)

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
        
        return reward*10.0
        
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
