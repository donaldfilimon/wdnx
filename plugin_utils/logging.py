import logging

# Structured logger factory
def get_logger(name=None):
    """Return a logger with a standardized format"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger 