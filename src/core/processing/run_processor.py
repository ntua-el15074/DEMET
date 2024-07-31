from config import ProcessingConfig
from processor import Processor

conf = ProcessingConfig()
processor = Processor(conf)

processor.process()
