from typing import List

import numpy as np
import pandas as pd

from utils.config import ConfigWAM
from utils.evaluation.compute_error import (
    _l2_error_along_axis,
    _rot_error,
    _task_space_error,
)
from utils.experiments.setpoint import EvalExperimentSetpoint


def eval_setpoint(
    df_metadata: pd.DataFrame,
    path_to_folder: str,
    end_error: int = 300,
    radian: bool = False,
    factor_integration_time: int = 1,
    pendulum_len: float = 0.3,
    time_horizon: int = 3000,
    eval_models: List[str] = ["standard", "standard_angled", "rotated", "human"],
    prefix: str = "",
):
    """Evaluate a given setpoint reach experiment.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Pandas df including all metadata (computational time, filenames, solver configs, etc.).
    only_end_error: int
        Only consider the last end_error points of the trajectory for the computation of
        the task space and angular error, the default is None and thus the whole trajectory.
    radian : bool, optional
        Should the angular errors be evaluated in radian, by default False and thus degrees
        are used.
    control_delay : int, optional
        Control delay (offset between mujoco and crocoddyl) used in this experiment, by default 0.
    time_horizon : int, optional
        How long are the trajectories of the original experiment.

    Returns
    -------
    x_err_df, rot_err_df, comp_time_df, conf_table
        Error in task space, rotational error, needed computational time, solver config parameters
    """
    # only read one specific experiment
    metadata = _prefilter_metadata(
        df_metadata, factor_integration_time=factor_integration_time
    )

    join_idx = ["dir", "radius", "orientation"]
    # start with blank df to join the results
    df = pd.DataFrame(columns=join_idx)

    for model_name in eval_models:
        mpc_times = [5, 10, 20]
        # define the evaluation experiment
        model = ConfigWAM(model_name, pendulum_len=pendulum_len)
        print("x_end_world: ", model.get_pend_end_world())
        experiment = EvalExperimentSetpoint(
            time_horizon, start_point=model.get_pend_end_world()
        )
        res = _eval_model(
            metadata,
            model_name,
            mpc_times,
            end_error,
            join_idx=join_idx,
            experiment=experiment,
            prefix=prefix,
            path_to_folder=path_to_folder,
            radian=radian,
        )
        df = pd.merge(df, res, on=join_idx, how="outer")
    x_err_df = df[join_idx + [col for col in df.columns if col.startswith("x")]]
    rot_err_df = df[join_idx + [col for col in df.columns if col.startswith("rot")]]
    comp_time_df = df[
        join_idx + [col for col in df.columns if col.startswith(prefix + "comp_time")]
    ]
    conf_table = _config_table(metadata, prefix=prefix)
    return x_err_df, rot_err_df, comp_time_df, conf_table


def _prefilter_metadata(df: pd.DataFrame, factor_integration_time: int = 1):
    """Filter for control_delay"""
    return df[(df["factor_integration_time"] == factor_integration_time)]


def _config_table(
    metadata: pd.DataFrame,
    prefix: str,
):

    config_cols = [
        "model_name",
        prefix + "u_pen",
        prefix + "x_pen",
        prefix + "rot_pen",
        prefix + "state_bound_pen",
        prefix + "state_pen",
        prefix + "q_pen",
        prefix + "v_pen",
    ]
    df_config = metadata[config_cols].groupby(["model_name"]).first()
    # model-types should be the columns of the table, replace NaN value with zeros
    df_config = df_config.transpose().fillna(0)
    return df_config


def _eval_model(
    metadata: pd.DataFrame,
    model_name: str,
    mpc_times: List[int],
    end_error: int,
    experiment: EvalExperimentSetpoint,
    prefix: str,
    path_to_folder: str,
    join_idx: List[str] = ["dir", "radius", "orientation"],
    radian: bool = False,
) -> pd.DataFrame:
    # evaluate one model for all values of mpc_horizon -> result is a joinable df

    # start with first one, join the rest
    mpc_horizon = mpc_times[0]
    df = _col_to_error(
        metadata,
        model_name,
        mpc_horizon,
        end_error,
        experiment=experiment,
        prefix=prefix,
        path_to_folder=path_to_folder,
        join_idx=join_idx,
        radian=radian,
    )
    for mpc_horizon in mpc_times[1:]:
        df_to_join = _col_to_error(
            metadata,
            model_name,
            mpc_horizon,
            end_error,
            experiment=experiment,
            prefix=prefix,
            path_to_folder=path_to_folder,
            join_idx=join_idx,
            radian=radian,
        )
        df = pd.merge(df, df_to_join, on=join_idx, how="left")

    df.columns = [
        "{}{}".format(
            col,
            "" if col in join_idx else "_{}".format(model_name),
        )
        for col in df.columns
    ]
    return df


def _col_to_error(
    metadata: pd.DataFrame,
    model_name: str,
    mpc_horizon: int,
    end_error: int,
    experiment: EvalExperimentSetpoint,
    prefix: str,
    path_to_folder: str,
    join_idx: List[str] = ["dir", "radius", "orientation"],
    radian: bool = False,
):
    return _entry_to_error(
        _read_db_col(metadata, model_name, mpc_horizon, join_idx),
        mpc_horizon,
        end_error,
        prefix=prefix,
        experiment=experiment,
        path_to_folder=path_to_folder,
        radian=radian,
    )


def _read_db_col(
    metadata: pd.DataFrame,
    model_name: str,
    mpc_horizon: int,
    join_idx: List[str] = ["dir", "radius", "orientation"],
) -> pd.DataFrame:
    df = metadata
    return df[
        (df["model_name"] == model_name) & (df["mpc_horizon"] == mpc_horizon)
    ].sort_values(join_idx)


def _entry_to_error(
    df_col: pd.DataFrame,
    mpc_time: int,
    end_error: int,
    experiment: EvalExperimentSetpoint,
    prefix: str,
    path_to_folder: str,
    radian: bool = False,
) -> pd.DataFrame:

    df_col["x_err"] = df_col.apply(
        lambda row: _l2_error_along_axis(
            _task_space_error(
                _load_single_entry(path_to_folder, row["x"], end_error),
                experiment.get_target(row["radius"], row["orientation"], row["dir"])[
                    -end_error:, :
                ],
            )
        ),
        axis=1,
    )
    df_col["rot_err"] = df_col.apply(
        lambda row: _l2_error_along_axis(
            _rot_error(
                _load_single_entry(path_to_folder, row["x"], end_error),
                _load_single_entry(path_to_folder, row["x_pend_beg"], end_error),
                radian=radian,
            ),
            use="abs",
        ),
        axis=1,
    )

    # rename cols to join them later on
    curr_names = ["x_err", "rot_err", prefix + "comp_time"]
    renamed_cols = [col + "_{}".format(mpc_time) for col in curr_names]
    rename = dict(zip(curr_names, renamed_cols))
    return df_col.rename(columns=rename)[
        ["dir", "radius", "orientation"] + renamed_cols
    ]


def _load_single_entry(
    data_meta_path: str, filename: str, end_error: int = None
) -> np.array:
    traj = np.load(data_meta_path + filename + ".npy")
    if end_error is None:
        return traj
    if traj.shape[0] < traj.shape[1]:
        return traj[:, -end_error:]
    return traj.T[:, -end_error:].T
