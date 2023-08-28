import time
from functools import partial, wraps

import rospy


def ros_log_time(func=None, get_time: bool = True):
    if func is None:
        return partial(ros_log_time, get_time=get_time)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            start = time.time()
            return_value = func(*args, **kwargs)
            time_needed = time.time() - start
            rospy.loginfo("\n{} took {:.4f}s\n".format(func.__name__, time_needed))
            if not get_time:
                return return_value
            return return_value, time_needed
        except Exception as exc:
            rospy.logfatal("func {} raised exception {}".format(func.__name__, exc))
            raise exc

    return wrapper
