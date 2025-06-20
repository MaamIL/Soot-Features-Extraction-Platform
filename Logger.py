import logging
import colorlog

class CustomLogger:
    """
    Custom logger class to handle logging with color and file output.
    """
    def __init__(self, log_filename, class_name):
        """
        Initialize the logger with a specified filename and class name.
        Args:
            log_filename (str): The filename where logs will be saved.
            class_name (str): The name of the class for which the logger is being created.
        """
        self.__class__name__ = class_name
        self.logger = logging.getLogger(self.__class__name__)
        self.logger.setLevel(logging.INFO)
        
        # Check if handlers are already added to avoid duplicate logs
        if not self.logger.handlers:
            # Create handlers
            stream_handler = colorlog.StreamHandler()
            file_handler = logging.FileHandler(log_filename)

            # Set level and format for handlers
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(levelname)s - %(class_name)s - %(message)s",
                datefmt=None,
                reset=True,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'white',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
                # secondary_log_colors={
                #     'class_name': {
                #         '__main__': 'white',
                #         'PowerPointVisual': 'blue',
                #         'FlameDataset': 'magenta',
                #         'Config': 'green',
                #         # Add more class names and colors as needed
                #     }
                # }
            )
            stream_handler.setFormatter(formatter)

            self.file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            file_handler.setFormatter(self.file_formatter)
            
            # Add handlers to the logger
            self.logger.addHandler(stream_handler)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        """
        Returns a LoggerAdapter with the class name included in log records.
        Returns:
            logging.LoggerAdapter: Logger adapter with class name.
        """
        logger = logging.LoggerAdapter(self.logger, {'class_name': self.__class__name__})
        return logger