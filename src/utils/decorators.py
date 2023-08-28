import logging
import os
import time
from datetime import timedelta
from functools import partial, wraps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def log_time(func=None, get_time: bool = True, time_in_mins: bool = False):
    if func is None:
        return partial(log_time, get_time=get_time)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            start = time.time()
            return_value = func(*args, **kwargs)
            time_needed = time.time() - start
            if time_in_mins:
                logger.info(
                    "{} took {}".format(func.__name__, timedelta(seconds=time_needed))
                )
            else:
                logger.info("{} took {:.4f}s".format(func.__name__, time_needed))
            if not get_time:
                return return_value
            return return_value, time_needed
        except Exception as exc:
            logger.exception("func {} raised exception {}".format(func.__name__, exc))
            raise exc

    return wrapper


def remove_mj_logs(func):
    def wraps(*args, **kwargs):
        _delete_mj_logs(folder="./")
        return func(*args, **kwargs)

    return wraps


def _delete_mj_logs(folder="./"):
    filename = folder + "MUJOCO_LOG.TXT"
    if os.path.exists(filename):
        os.remove(filename)
