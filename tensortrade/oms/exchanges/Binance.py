import logging
from decimal import Decimal

from tensortrade.core import Clock
from tensortrade.oms.wallets import Wallet
from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.orders import Order, Trade, TradeType, TradeSide

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from binance.enums import *

class Binance():

    def __init__(self,
                 api_key,
                 api_secret,
                 symbol: str,
                 base_instrument: str,
                 useTestNet: bool = False):
        
        self.client = Client(api_key, api_secret, testnet=useTestNet)

        # print(self.client.get_account())

        # if not useTestNet:
        #     self.fees = self.client.get_trade_fee(symbol=symbol.upper()+base_instrument.upper())

        self.asset_balance = 0
        self.base_instrument_balance = 10000.0
        # self.asset_balance = self.client.get_asset_balance(asset=symbol.upper())["free"]
        # self.base_instrument_balance = self.client.get_asset_balance(asset=base_instrument.upper())["free"]

        # print('account balance, {} = '.format(symbol), self.asset_balance)
        # print('account balance, {} = '.format(base_instrument), self.base_instrument_balance)
        
        self.pair = symbol+base_instrument
        
    def get_historical_klines(self,
                              trading_pair: str,
                              interval,
                              from_date: str = None,
                              to_date: str = None
                              ) -> list :

        return self.client.get_historical_klines(trading_pair, interval, from_date, to_date)


    
        
    def execute_buy_order(order: 'Order',
                          base_wallet: 'Wallet',
                          quote_wallet: 'Wallet',
                          current_price: float,
                          options: 'ExchangeOptions',
                          clock: 'Clock') -> 'Trade':
        """Executes a buy order on the exchange.

        Parameters
        ----------
        order : `Order`
            The order that is being filled.
        base_wallet : `Wallet`
            The wallet of the base instrument.
        quote_wallet : `Wallet`
            The wallet of the quote instrument.
        current_price : float
            The current price of the exchange pair.
        options : `ExchangeOptions`
            The exchange options.
        clock : `Clock`
            The clock for the trading process..

        Returns
        -------
        `Trade`
            The executed trade that was made.
        """
        if order.type == TradeType.LIMIT and order.price < current_price:
            return None

        filled = order.remaining.contain(order.exchange_pair)

        if order.type == TradeType.MARKET:
            scale = order.price / max(current_price, order.price)
            filled = scale * filled

        commission = options.commission * filled
        quantity = filled - commission

        if commission.size < Decimal(10) ** -quantity.instrument.precision:
            logging.warning("Commission is less than instrument precision. Canceling order. "
                            "Consider defining a custom instrument with a higher precision.")
            order.cancel("COMMISSION IS LESS THAN PRECISION.")
            return None

        transfer = Wallet.transfer(
            source=base_wallet,
            target=quote_wallet,
            quantity=quantity,
            commission=commission,
            exchange_pair=order.exchange_pair,
            reason="BUY"
        )

        trade = Trade(
            order_id=order.id,
            step=clock.step,
            exchange_pair=order.exchange_pair,
            side=TradeSide.BUY,
            trade_type=order.type,
            quantity=transfer.quantity,
            price=transfer.price,
            commission=transfer.commission
        )
        
        order = self.client.create_test_order(
            symbol=order.exchange_pair,
            side=SIDE_BUY,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=transfer.quantity,
            price=transfer.price)
        
        return trade


    def execute_sell_order(order: 'Order',
                           base_wallet: 'Wallet',
                           quote_wallet: 'Wallet',
                           current_price: float,
                           options: 'ExchangeOptions',
                           clock: 'Clock') -> 'Trade':
        """Executes a sell order on the exchange.

        Parameters
        ----------
        order : `Order`
            The order that is being filled.
        base_wallet : `Wallet`
            The wallet of the base instrument.
        quote_wallet : `Wallet`
            The wallet of the quote instrument.
        current_price : float
            The current price of the exchange pair.
        options : `ExchangeOptions`
            The exchange options.
        clock : `Clock`
            The clock for the trading process..

        Returns
        -------
        `Trade`
            The executed trade that was made.
        """
        if order.type == TradeType.LIMIT and order.price > current_price:
            return None

        filled = order.remaining.contain(order.exchange_pair)

        commission = options.commission * filled
        quantity = filled - commission

        if commission.size < Decimal(10) ** -quantity.instrument.precision:
            logging.warning("Commission is less than instrument precision. Canceling order. "
                            "Consider defining a custom instrument with a higher precision.")
            order.cancel("COMMISSION IS LESS THAN PRECISION.")
            return None

        # Transfer Funds from Quote Wallet to Base Wallet
        transfer = Wallet.transfer(
            source=quote_wallet,
            target=base_wallet,
            quantity=quantity,
            commission=commission,
            exchange_pair=order.exchange_pair,
            reason="SELL"
        )

        trade = Trade(
            order_id=order.id,
            step=clock.step,
            exchange_pair=order.exchange_pair,
            side=TradeSide.SELL,
            trade_type=order.type,
            quantity=transfer.quantity,
            price=transfer.price,
            commission=transfer.commission
        )

        order = self.client.create_test_order(
            symbol=order.exchange_pair,
            side=SIDE_SELL,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=transfer.quantity,
            price=transfer.price)

        return trade


    def execute_order(order: 'Order',
                      base_wallet: 'Wallet',
                      quote_wallet: 'Wallet',
                      current_price: float,
                      options: 'Options',
                      clock: 'Clock') -> 'Trade':
        """Executes an order on the exchange.

        Parameters
        ----------
        order : `Order`
            The order that is being filled.
        base_wallet : `Wallet`
            The wallet of the base instrument.
        quote_wallet : `Wallet`
            The wallet of the quote instrument.
        current_price : float
            The current price of the exchange pair.
        options : `ExchangeOptions`
            The exchange options.
        clock : `Clock`
            The clock for the trading process..

        Returns
        -------
        `Trade`
            The executed trade that was made.
        """
        kwargs = {"order": order,
                  "base_wallet": base_wallet,
                  "quote_wallet": quote_wallet,
                  "current_price": current_price,
                  "options": options,
                  "clock": clock}

        if order.is_buy:
            trade = execute_buy_order(**kwargs)
        elif order.is_sell:
            trade = execute_sell_order(**kwargs)
        else:
            trade = None

        return trade
