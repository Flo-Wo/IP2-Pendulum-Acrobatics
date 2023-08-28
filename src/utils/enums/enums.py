from enum import Enum


class WarmstartType(str, Enum):
    no_warmstart = "none"
    quasi_static = "quasi_static"
    cache_last_result = "cache"


class BenchmarkTestSize(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"


class CompTimeNames:
    mpc_comp_time = "mpc_comp_time"
    ddp_comp_time = "ddp_comp_time"
