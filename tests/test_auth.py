import cira
from unittest.mock import patch, Mock


# Helper Mock class for the quote object
class MockQuote:
    def __init__(self, ask_price):
        self.ask_price = ask_price


def test_checking_keys_real_fail_by_default():
    """
    To check if keys are correct we need valid keys.
    By default, without environment variables or a key file,
    get_api_keys() will raise a ValueError, and check_keys() will return False.
    """
    assert not cira.auth.check_keys()


@patch("cira.auth.StockHistoricalDataClient")
def test_check_keys_success_mocked(MockStockClient):
    """
    Tests check_keys() successful path by mocking Alpaca API calls.
    """
    # Configure the mock client instance
    mock_client_instance = MockStockClient.return_value

    # Configure the mock quote object
    mock_quote = Mock()  # Using Mock directly for simplicity
    mock_quote.ask_price = 300.0

    # Configure get_stock_latest_quote to return the mock quote
    mock_client_instance.get_stock_latest_quote.return_value = {"SPY": mock_quote}

    # Mock get_api_keys to return dummy keys to prevent ValueError
    with patch("cira.auth.get_api_keys", return_value=("DUMMY_ID", "DUMMY_KEY")):
        assert cira.auth.check_keys() is True
