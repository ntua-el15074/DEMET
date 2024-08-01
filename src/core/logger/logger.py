import os
import time
from core.logger.config import LOGPATH, COLORS

class Logger:
    def __init__(self, to_file = False):
        self.to_file = to_file
        self.data = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.path = os.path.join(LOGPATH, 'log-' + self.data + '.log')
        self.colors = COLORS

    def __str__(self):
        if self.to_file:
            return 'Logging to file'

    def print(self, message, color = 'reset', bold = False):
        if bold:
            print(self.colors['bold'] + self.colors[color] + '[' + self.string_by_time() + ']:' + ' ' + self.colors['reset'] + message)
        else:
            print(self.colors[color] + '[' + self.string_by_time() + ']:' + ' ' + self.colors['reset'] + message)

    def log(self, message):
        if self.to_file:
            with open(self.path, 'a') as file:
                file.write('[' + self.string_by_time() + ']:' + ' ' + message + '\n')

    def print_and_log(self, message, color = 'reset', bold = False):
        if bold:
            print(self.colors['bold'] + self.colors[color] + '[' + self.string_by_time() + ']:' + ' ' + self.colors['reset'] + message)
        else:
            print(self.colors[color] + '[' + self.string_by_time() + ']:' + ' ' + self.colors['reset'] + message)
        if self.to_file:
            with open(self.path, 'a') as file:
                file.write('[' + self.string_by_time() + ']:' + ' ' + message + '\n')

    def string_by_time(self):
        return time.strftime('%H:%M:%S', time.localtime())


