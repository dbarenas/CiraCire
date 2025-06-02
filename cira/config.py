import logging

"""
 This files keeps the varibale that is used 
 for configuration cross cira. 
"""

# logging
IS_LOGGING = False
LOG_FILE = "./cira-log.csv"  # No trailing whitespace
LOGGING_LEVEL = logging.WARNING  # No trailing whitespace

# debugging
DEBUG = False

# paper trading
PAPER_TRADING = True

# data that will not very often change is cached
USE_CASHING = True
