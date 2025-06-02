import json
# import warnings # Unused
import os

from alpaca.data import StockHistoricalDataClient, StockLatestQuoteRequest

# from .portfolio import Portfolio # Unused


"""
This function let's you interact 
with the Alpaca trade API 
"""

# Private global variables for storing API keys and key file path
_KEY_FILE: str = ""  # No trailing whitespace
_APCA_API_KEY_ID: str = ""  # No trailing whitespace
_APCA_API_SECRET_KEY: str = ""  # No trailing whitespace


def set_api_keys(api_key_id: str, secret_key: str) -> None:
    """Sets the Alpaca API Key ID and Secret Key."""
    global _APCA_API_KEY_ID, _APCA_API_SECRET_KEY
    _APCA_API_KEY_ID = api_key_id
    _APCA_API_SECRET_KEY = secret_key
    # Unset key file path if direct keys are provided
    global _KEY_FILE
    _KEY_FILE = ""


def set_key_file_path(path: str) -> None:
    """Sets the path to the JSON file containing Alpaca API keys."""
    global _KEY_FILE
    _KEY_FILE = path
    # Unset direct keys if file path is provided
    global _APCA_API_KEY_ID, _APCA_API_SECRET_KEY
    _APCA_API_KEY_ID = ""
    _APCA_API_SECRET_KEY = ""


def get_api_keys():
    # No 'global' needed here as we are only reading module-level variables
    # global _KEY_FILE, _APCA_API_KEY_ID, _APCA_API_SECRET_KEY

    # Prioritize environment variables
    env_api_id = os.environ.get("APCA_ID")  # No W293 on this line (was line 49)
    env_api_key = os.environ.get("APCA_KEY")
    if env_api_id and env_api_key:
        return env_api_id, env_api_key
    # Then, check if _KEY_FILE is set and try to use it
    if _KEY_FILE:
        auth_header = authentication_header()  # Uses module-level _KEY_FILE
        if auth_header:
            header_api_id = auth_header.get("APCA-API-KEY-ID")  # E501 on original line 59 - no change here
            header_api_key = auth_header.get("APCA-API-SECRET-KEY")
            if header_api_id and header_api_key:
                return str(header_api_id), str(header_api_key)

    # Finally, use module-level _APCA_API_KEY_ID and _APCA_API_SECRET_KEY if set
    if _APCA_API_KEY_ID and _APCA_API_SECRET_KEY:
        return _APCA_API_KEY_ID, _APCA_API_SECRET_KEY

    # If no keys are found through any method, raise error
    url = "https://github.com/AxelGard/cira/wiki/Storing-the-Alpaca-API-key"
    # E501 on original line 65 - attempting to shorten
    msg = (
        "Alpaca market keys were not given or found. "
        "Please set them using environment variables, "
        "set_api_keys(), or set_key_file_path(). Docs: " + url
    )
    raise ValueError(msg)


def check_keys() -> bool:
    try:
        api_id_to_use, api_key_to_use = get_api_keys()
        # get_api_keys now raises ValueError if keys are not found/configured.
        stock_client = StockHistoricalDataClient(api_id_to_use, api_key_to_use)
        perms = StockLatestQuoteRequest(symbol_or_symbols="SPY")
        # Call made to ensure keys work, result not needed
        stock_client.get_stock_latest_quote(perms)["SPY"].ask_price
        return True
    except Exception:  # Changed bare except to except Exception (E261 fix)
        return False


def authentication_header():
    """get's key and returns key in json format"""
    # No 'global' needed here as we are only reading module-level _KEY_FILE
    # global _KEY_FILE  # Uses the private global _KEY_FILE (E261 fix for comment)
    if not _KEY_FILE:
        return None  # Or raise error, but get_api_keys handles missing file by moving on (E261 fix for comment)
    try:
        with open(_KEY_FILE, "r") as file:  # E501 on original line 85 - no change here
            header = json.load(file)
        return header
    except FileNotFoundError:
        # This case should ideally be handled or logged. (E501 on original line 92 - no change here)
        # get_api_keys currently moves to the next method if _KEY_FILE is set but invalid.
        return None
