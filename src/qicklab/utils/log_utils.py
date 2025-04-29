import logging

def configure_logging(log_file):
    ''' We need to create a custom logger and disable propagation like this
    to remove the logs from the underlying qick from saving to the log file for RR'''

    rr_logger = logging.getLogger("custom_logger_for_rr_only")
    rr_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    rr_logger.addHandler(file_handler)
    rr_logger.propagate = False  # dont propagate logs from underlying qick package

    return rr_logger