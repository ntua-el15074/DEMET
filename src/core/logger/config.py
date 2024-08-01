import os

COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'bold': '\033[1m',
    'reset': '\033[0m'
}

DEMET_PATH = '/Users/peteraugerinos/Coding/NTUA Dev/Thesis Umbrella/DEMET/'
LOGPATH = DEMET_PATH + 'logs/'
if (os.path.exists(LOGPATH) == False):
    os.makedirs(LOGPATH)
else:
    os.makedirs(LOGPATH, exist_ok=True)
