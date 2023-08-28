import logging
from typing import Dict

import numpy as np
import pandas as pd

from utils import DataBase
from utils.data_handling.helper import (
    already_computed,
    combine_indices,
    combine_metadata,
    generate_base_str,
    load_row,
    read_db,
    save_db,
)

"""
previous call:

* config_model: ConfigModelBase,
* penalties: PenaltyBase,
* experiment: ExperimentBase,
* results: SolverResultsBase,
* add_data: DataBase,
"""


class DataSaver:
    def __init__(self, path_metadata: str, file_folder: str):
        self.path_metadata = path_metadata
        self.file_folder = file_folder

    def save_data(self, *args: DataBase):
        self.base_str = generate_base_str(*args)
        add_metadata = self._save_files(*args)
        table_idx = combine_indices(add_metadata, *args)
        new_metadata = combine_metadata(add_metadata, *args)
        self._save_metadata(table_idx, new_metadata)

    def _save_metadata(self, table_idx: dict, new_metadata: dict):
        # load the database
        db = read_db(self.path_metadata)

        if db is not None:
            # if already_computed(db, table_idx):
            #     logging.info("Row already existed, is overwritten.")
            #     db = db.drop(load_row(db, table_idx).index)
            db_raw = db.to_dict("records")
            db_raw.append(new_metadata)
        else:
            db_raw = [new_metadata]
        db_df = pd.DataFrame.from_dict(db_raw)
        # save the new db
        save_db(self.path_metadata, db_df)

    def _save_files(self, *args: DataBase) -> Dict[str, str]:
        file_paths = {}
        for data_instance in args:
            for suffix, data in data_instance.save_to_file().items():
                # create unique filename: base_str + suffix
                filename = self.base_str + suffix
                # save the file
                np.save(self.file_folder + filename + ".npy", data)
                # add the filename to the filenames dict
                # --> will be added to the metadata table
                file_paths[suffix] = filename

        return file_paths

    # helper functions to load and save the pandas metadata table
    def _read_db(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.path_metadata)
        except:
            logging.info("Database not found!")
            return None

    def _save_db(self, db: pd.DataFrame) -> None:
        db.to_csv(self.path_metadata, index=False)
