import logging
import nqcpresonator._version


__version__ = nqcpresonator._version.__version__



logger = logging.getLogger(__name__)
logger.info(f'Imported nqcpresonatorversion: {__version__}')
