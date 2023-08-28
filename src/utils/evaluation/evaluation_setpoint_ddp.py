from copy import copy
from typing import List

import numpy as np
import pandas as pd

from utils.config import ConfigWAM
from utils.evaluation.compute_error import _l2_error_along_axis, _task_space_error
from utils.evaluation.evaluation_setpoint import _config_table, _prefilter_metadata
from utils.experiments.setpoint import EvalExperimentSetpoint
from utils.pin_helper import angles_traj, task_space_traj


def eval_ddp_precomputed(
    df_metadata: pd.DataFrame,
    path_to_folder: str,
    end_error: int = 300,
    radian: bool = False,
    factor_integration_time: int = 1,
    time_horizon: int = 3000,
    eval_models: List[str] = ["standard", "standard_angled", "rotated", "human"],
    prefix: str = "",
    pend_end_frame_idx: int = 24,
    pend_beg_frame_idx: int = 22,
    pendulum_len: float = 0.3,
):
    metadata = _prefilter_metadata(
        df_metadata, factor_integration_time=factor_integration_time
    )

    join_idx = ["dir", "radius", "orientation"]
    df = pd.DataFrame(columns=join_idx)

    for model_name in eval_models:
        # define the evaluation experiment
        model = ConfigWAM(model_name, pendulum_len=pendulum_len)
        print("x_end_world: ", model.get_pend_end_world())
        experiment = EvalExperimentSetpoint(
            time_horizon, start_point=model.get_pend_end_world()
        )
        res = _eval_model_ddp(
            metadata,
            model_name,
            end_error,
            experiment,
            prefix,
            pin_model=copy(model.pin_model),
            pin_data=copy(model.pin_data),
            pend_end_frame_idx=pend_end_frame_idx,
            pend_beg_frame_idx=pend_beg_frame_idx,
            path_to_folder=path_to_folder,
            join_idx=join_idx,
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


def _eval_model_ddp(
    metadata: pd.DataFrame,
    model_name: str,
    end_error: int,
    experiment: EvalExperimentSetpoint,
    prefix: str,
    pin_model,
    pin_data,
    pend_end_frame_idx: int,
    pend_beg_frame_idx: int,
    path_to_folder: str,
    join_idx: List[str] = ["dir", "radius", "orientation"],
    radian: bool = False,
) -> pd.DataFrame:
    df = _col_to_error(
        metadata,
        model_name,
        end_error,
        experiment=experiment,
        prefix=prefix,
        pin_model=pin_model,
        pin_data=pin_data,
        pend_end_frame_idx=pend_end_frame_idx,
        pend_beg_frame_idx=pend_beg_frame_idx,
        path_to_folder=path_to_folder,
        join_idx=join_idx,
        radian=radian,
    )
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
    end_error: int,
    experiment: EvalExperimentSetpoint,
    prefix: str,
    pin_model,
    pin_data,
    pend_end_frame_idx: int,
    pend_beg_frame_idx: int,
    path_to_folder: str,
    join_idx: List[str] = ["dir", "radius", "orientation"],
    radian: bool = False,
):
    return _entry_to_error(
        _read_db_col(metadata, model_name, join_idx),
        end_error=end_error,
        experiment=experiment,
        prefix=prefix,
        pin_model=pin_model,
        pin_data=pin_data,
        pend_end_frame_idx=pend_end_frame_idx,
        pend_beg_frame_idx=pend_beg_frame_idx,
        path_to_folder=path_to_folder,
        radian=radian,
    )


def _read_db_col(
    metadata: pd.DataFrame,
    model_name: str,
    join_idx: List[str] = ["dir", "radius", "orientation"],
) -> pd.DataFrame:
    df = metadata
    return df[(df["model_name"] == model_name) & (df["mpc_horizon"] == 5)].sort_values(
        join_idx
    )


def _entry_to_error(
    df_col: pd.DataFrame,
    end_error: int,
    experiment: EvalExperimentSetpoint,
    prefix: str,
    pin_model,
    pin_data,
    pend_end_frame_idx: int,
    pend_beg_frame_idx: int,
    path_to_folder: str,
    radian: bool = False,
) -> pd.DataFrame:

    df_col["x_err"] = df_col.apply(
        lambda row: _l2_error_along_axis(
            _task_space_error(
                _load_single_entry(
                    path_to_folder,
                    row[prefix + "states"],
                    pin_model,
                    pin_data,
                    pend_beg_frame_idx=pend_end_frame_idx,
                    pend_end_frame_idx=pend_end_frame_idx,
                    end_error=end_error,
                    angle=False,
                ),
                experiment.get_target(row["radius"], row["orientation"], row["dir"])[
                    -end_error:, :
                ],
            )
        ),
        axis=1,
    )
    df_col["rot_err"] = df_col.apply(
        lambda row: _l2_error_along_axis(
            _load_single_entry(
                path_to_folder,
                row[prefix + "states"],
                pin_model,
                pin_data,
                pend_beg_frame_idx=pend_beg_frame_idx,
                pend_end_frame_idx=pend_end_frame_idx,
                radian=radian,
                end_error=end_error,
                angle=True,
            ),
            use="abs",
        ),
        axis=1,
    )

    # rename cols to join them later on
    curr_names = ["x_err", "rot_err", prefix + "comp_time"]
    return df_col[["dir", "radius", "orientation"] + curr_names]


def _load_single_entry(
    data_meta_path: str,
    filename: str,
    pin_model,
    pin_data,
    pend_beg_frame_idx: int,
    pend_end_frame_idx: int,
    radian: bool = False,
    end_error: int = None,
    angle: bool = False,
) -> np.array:
    traj_raw = np.load(data_meta_path + filename + ".npy")
    q_list = np.array([x[:6] for x in traj_raw[1:]])
    q_dot_list = np.array([x[:6] for x in traj_raw[1:]])
    if angle:
        traj = angles_traj(
            pin_model,
            pin_data,
            q_list,
            upper_frame_id=pend_end_frame_idx,
            lower_frame_id=pend_beg_frame_idx,
            radian=radian,
        )
    else:
        traj = task_space_traj(
            pin_model,
            pin_data,
            q_list,
            frame_idx=pend_end_frame_idx,
        )
    if end_error is None:
        return traj
    if traj.ndim > 1:
        if traj.shape[0] < traj.shape[1]:
            return traj[:, -end_error:]
        return traj.T[:, -end_error:].T
    else:
        return traj[-end_error:]
