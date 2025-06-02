# Alpaca
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus  # OrderType removed
# Bar removed
# OrderSide, TimeInForce removed
# TimeFrame removed
# LimitOrderRequest, StopLimitOrderRequest removed
from alpaca.trading.client import TradingClient


# stock
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
# MarketOrderRequest removed
from alpaca.data.live import StockDataStream


# cira
from .asset import Asset
from . import auth
from . import config
# util removed
# log removed


class Stock(Asset):
    def __init__(self, symbol: str) -> None:
        """Exchange for trading stocks"""
        APCA_ID, APCA_SECRET = auth.get_api_keys()
        self.symbol = symbol
        self.live_client = StockDataStream(APCA_ID, APCA_SECRET)
        self.history = StockHistoricalDataClient(APCA_ID, APCA_SECRET)
        self.trade = TradingClient(APCA_ID, APCA_SECRET, paper=config.PAPER_TRADING)
        self.latest_quote_request = StockLatestQuoteRequest
        self.bars_request = StockBarsRequest

    def price(self) -> float:
        """gets the asking price of the symbol"""
        perms = self.latest_quote_request(symbol_or_symbols=self.symbol)
        return float(self.history.get_stock_latest_quote(perms)[self.symbol].ask_price)

    @classmethod
    def get_all_assets(self):
        APCA_ID, APCA_SECRET = auth.get_api_keys()
        trade = TradingClient(APCA_ID, APCA_SECRET, paper=config.PAPER_TRADING)
        search_params = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE
        )
        return [a.symbol for a in trade.get_all_assets(search_params)]
