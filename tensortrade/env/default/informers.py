import pandas as pd
from tensortrade.env.generic import Informer, TradingEnv


class TensorTradeInformer(Informer):

    def __init__(self) -> None:
        super().__init__()

    def info(self, env: 'TradingEnv') -> dict:
        return {
            'step': self.clock.step,
            'df_price_history': pd.DataFrame(env.observer.renderer_history),
            'df_performance': pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index'),
            'list_trade_object': env.action_scheme.broker.trades.values()
            
        }
