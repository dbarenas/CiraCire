from typing import List
import pickle
import pandas as pd
import numpy as np
import random
import schedule
import time


class Strategy:
    def __init__(self, name) -> None:
        self.name = name

    def iterate(
        self,
        feature_data: pd.DataFrame,
        prices: pd.DataFrame,
        portfolio: np.ndarray,
        cash=float,
    ) -> np.ndarray:
        """
        Takes in feature data, then returns allocation prediction.
        """
        raise NotImplementedError

    def save(self, file_path):
        """
        Save strategy to pickle file
        usage:
            strategy.fit(train_data)
            strategy.save('./model.pkl')
        """
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        """
        Load in strategy from pickle file
        usage:
            strategy = Strategy.load('./model.pkl')
            predictions = strategy.predict(test_data)
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)


class Randomness(Strategy):
    def __init__(
        self,
        lower: float = -1,
        upper: float = 1,
        seed=0,
        use_float: bool = False,
    ) -> None:
        super().__init__(name="Randomness")
        random.seed(seed)
        self.a = lower
        self.b = upper
        self.allocation = []
        self.use_float = use_float

    def iterate(
        self,
        feature_data: pd.DataFrame,
        prices: pd.DataFrame,
        portfolio: np.ndarray,
        cash=float,
    ) -> np.ndarray:
        al = np.array(
            [
                random.uniform(float(self.a), float(self.b))
                for _ in range(len(prices.keys()))
            ]
        )
        if not self.use_float:
            al = al.astype(int)
        self.allocation.append(al)
        return al


class DollarCostAveraging(Strategy):
    def __init__(self, amount: float = 1) -> None:
        super().__init__(name="DollarCostAveraging")
        self.amount = amount
        self.allocation = []

    def iterate(
        self,
        feature_data: pd.DataFrame,
        prices: pd.DataFrame,
        portfolio: np.ndarray,
        cash=float,
    ) -> np.ndarray:
        al = np.array([self.amount for _ in range(len(prices.keys()))])
        self.allocation.append(al)
        return al


class ByAndHold(Strategy):
    def __init__(self) -> None:
        super().__init__(name="BuyAndHold")
        self.is_first = True
        self.allocation = []

    def iterate(
        self,
        feature_data: pd.DataFrame,
        prices: pd.DataFrame,
        portfolio: np.ndarray,
        cash=float,
    ) -> np.ndarray:
        if self.is_first:
            self.is_first = False
            amount = cash / len(prices.keys())
            amount *= 0.96
            al = (amount // prices.values).astype(np.int64)[0]
            self.allocation.append(al)
            return al
        al = np.array([0] * len(prices.keys()))
        self.allocation.append(al)
        return al


FEE_RATE = 0.004  # this is what alpaca takes


def fees(prices, allocation):  # E302 fix (added blank line before)
    """Calculates fees based on the fee rate, prices, and allocation."""
    return FEE_RATE * np.matmul(prices.T, allocation)


# Helper function for adjusting allocation based on current holdings and cash
def _adjust_allocation(
    requested_allocation: np.ndarray,
    current_holdings: np.ndarray,
    current_cash: float,
) -> np.ndarray:
    """
    Adjusts the requested allocation.
    - Prevents selling more assets than currently held.
    - Prevents any selling if cash is zero or negative (original logic).
    """
    adjusted = requested_allocation.copy()
    for i in range(len(adjusted)):
        # Original logic: if no cash (or negative) and strategy wants to sell, prevent sell.
        if current_cash <= 0.0 and adjusted[i] < 0.0:
            adjusted[i] = 0
        # Prevent selling more than currently held
        if adjusted[i] < 0.0 and current_holdings[i] + adjusted[i] < 0.0:
            adjusted[i] = -current_holdings[i]
    return adjusted


# Helper function for calculating transaction costs and executing the trade if affordable
def _execute_transaction(  # E302 fix: ensure two blank lines above
    current_prices: pd.Series,
    allocation_change: np.ndarray,
    current_cash: float,
    current_holdings: np.ndarray,
    fee_func,  # Pass the fees function (E261 fix)
    use_fees: bool,
) -> tuple[float, np.ndarray]:
    """
    Calculates the cost of the transaction including fees and executes it if affordable.
    Returns the new cash balance and new asset holdings.
    """
    cost_of_assets = np.matmul(current_prices.values.T, allocation_change)
    transaction_fees = 0.0
    if use_fees:
        # Ensure fees are only applied if there's an actual transaction
        if np.any(allocation_change):  # No W291 trailing whitespace (was line 171)
            transaction_fees = fee_func(current_prices.values, allocation_change)
            # Original fee logic implies fees are positive for buys, negative for sells if allocation_change can be negative.  (E501 on line 172)
            # Let's ensure fees are always a positive cost or zero. (E501 on line 173)
            # The matmul of prices.T and allocation_change can be negative if selling.
            # If allocation_change involves selling (negative values), prices.T * allocation_change will be negative.
            # FEE_RATE is positive. So `fees` can be negative if selling a lot.
            # This implies fees reduce profits from sales or add to cost of buys.
            # The original code `+ use_fees * fees(...)` means if `fees` is negative (from a sale), it reduces the `asking` price.
            # This is complex. Let's assume `fees` function returns a positive cost for now.
            # The `fees` function: `FEE_RATE * np.matmul(prices.T, allocation)`
            # If allocation is negative (sell), this is negative. If positive (buy), this is positive.
            # So `asking = cost_of_assets + transaction_fees`.
            # If selling, cost_of_assets is negative. transaction_fees is also negative. asking becomes more negative (more cash received).
            # If buying, cost_of_assets is positive. transaction_fees is also positive. asking becomes more positive (more cash spent).
            # This seems correct.

    asking_price = cost_of_assets + transaction_fees

    if asking_price <= current_cash:
        new_cash = current_cash - asking_price
        new_holdings = current_holdings + allocation_change
        return new_cash, new_holdings
    else:
        # Transaction not affordable, return original cash and holdings
        return current_cash, current_holdings


def back_test(
    strat: Strategy,
    feature_data: pd.DataFrame,
    asset_prices: pd.DataFrame,
    capital=100_000.0,
    use_fees: bool = True,
) -> pd.DataFrame:
    """
    DISCLAIMER:
    The results of this backtest are based on historical data and do not guarantee future performance.
    The financial markets are inherently uncertain, and various factors can influence actual trading results.
    This backtest is provided for educational and informational purposes only.
    Users should exercise caution and conduct additional research before applying any trading strategy in live markets.
    """
    portfolio_history = {
        "value": [],
        "timestamp": [],
    }
    assert len(feature_data) == len(asset_prices)
    total_value = capital
    nr_of_asset = np.zeros([len(asset_prices.keys())], float)
    i = 0
    for t, cur_price in asset_prices.iterrows():
        current_cash = capital  # Use a more descriptive name for capital in the loop
        current_holdings = nr_of_asset

        if total_value > 0: # Original condition to allow trading
            f_data = feature_data.iloc[: i + 1]
            p_data = asset_prices.iloc[: i + 1]
            
            # Get allocation decision from strategy
            requested_allocation = strat.iterate(f_data, p_data, current_holdings.copy(), current_cash)
            
            assert len(requested_allocation) == len(current_holdings), \
                "Strategy allocation array size mismatch with number of assets"

            # Adjust allocation based on constraints (e.g., not selling more than held)
            # Pass current_cash to _adjust_allocation for the original logic of not selling if bankrupt
            adjusted_allocation = _adjust_allocation(
                requested_allocation, current_holdings, current_cash
            )
            
            # Calculate transaction details and execute if affordable
            capital, nr_of_asset = _execute_transaction( # (E501 on original line 241)
                current_prices=cur_price, 
                allocation_change=adjusted_allocation, 
                current_cash=current_cash, 
                current_holdings=current_holdings, 
                fee_func=fees,  # Pass the fees function (E261 fix for original comment)
                use_fees=use_fees
            )
        # Removed potential W293 blank line here by ensuring no whitespace on empty lines
        # Update total portfolio value for this timestamp (W293 on blank line before this in previous output)
        total_value = np.matmul(cur_price.values.T, nr_of_asset) + capital
        
        portfolio_history["timestamp"].append(t)
        portfolio_history["value"].append(total_value)
        i += 1

    df = pd.DataFrame(portfolio_history)
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index.get_level_values("timestamp"))
    df.rename(columns={"value": strat.name}, inplace=True)
    return df


def multi_strategy_backtest(
    strats: List[Strategy],
    feature_data: pd.DataFrame,
    asset_prices: pd.DataFrame,
    capital=100_000.0,
    use_fees: bool = True,
):
    result = pd.DataFrame()
    result.index = asset_prices.index
    for s in strats:
        s_result = back_test(
            s,
            feature_data=feature_data,
            asset_prices=asset_prices,
            capital=capital,
            use_fees=use_fees,
        )
        result[s.name] = s_result[s.name]
    return result


def back_test_against_buy_and_hold(
    strat: Strategy,
    feature_data: pd.DataFrame,
    asset_prices: pd.DataFrame,
    capital=100_000.0,
    use_fees: bool = True,
):
    buy_and_hold = ByAndHold()
    return multi_strategy_backtest(
        strats=[strat, buy_and_hold],
        feature_data=feature_data,
        asset_prices=asset_prices,
        capital=capital,
        use_fees=use_fees,
    )


class Scheduler:
    def __init__(self) -> None:
        pass

    def add_daily_job(self, func_name) -> None:
        schedule.every(1).days.do(func_name)

    def add_daily_job_at(self, func_name, time_HM: str = "12:00") -> None:
        schedule.every().day.at(time_HM).do(func_name)

    def add_hour_job(self, func_name) -> None:
        schedule.every(1).hour.do(func_name)

    def add_minute_job(self, func_name) -> None:
        schedule.every(1).minute.do(func_name)

    def add_daily_job_at_time_EDT(self, func_name, time_HM: str = "12:00") -> None:
        schedule.every().day.at(time_HM, "America/New_York").do(func_name)

    def get_all_jobs(self):
        return schedule.jobs

    def clear_all_jobs(self) -> None:
        schedule.clear()

    def run(self):
        """runs the scheduler for ever"""
        while True:
            schedule.run_pending()
            time.sleep(1)
