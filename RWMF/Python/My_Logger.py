import logging
import os.path as path
from datetime import datetime

class Logger:

    # Create logger

    def __init__(self,dir_output,logger = None,formatter = None):
        if formatter:
            self.logFormatter = formatter
        else:
            self.logFormatter = logging.Formatter("%(levelname)s - [%(asctime)s] - %(message)s", datefmt = '%m/%d %H:%M:%S')
        
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            
            # create console handler and set level to info
            cons_handler = logging.StreamHandler()
            cons_handler.setLevel(logging.INFO)
            cons_handler.setFormatter(self.logFormatter)
            self.logger.addHandler(cons_handler)
            
            now = datetime.now()
            
            # create error file handler and set level to error
            err_file = now.strftime('err_%m%d_%H%M.log')
            err_handler = logging.FileHandler(path.join(dir_output,err_file))
            err_handler.setLevel(logging.ERROR)
            cons_handler.setFormatter(self.logFormatter)
            self.logger.addHandler(cons_handler)
            
            # create debug file handler and set level to debug
            debug_file = now.strftime('debug__%m%d_%H%M.log')
            debug_handler = logging.FileHandler(path.join(dir_output,debug_file))
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(self.logFormatter)
            self.logger.addHandler(debug_handler)

