import numpy as np
from utils import DataBase
from utils.data_handling.helper import (
    generate_base_str,
    read_db,
    load_row,
    combine_metadata,
)


class DataLoader:
    def __init__(self, path_metadata: str, file_folder: str):
        self.path_metadata = path_metadata
        self.file_folder = file_folder

    def load_data(self, *args: DataBase):
        self.base_str = generate_base_str(*args)

        db = read_db(self.path_metadata)
        # the combined metadata entries are the index of the table
        # (filenames are not needed)
        row_dict = self._row_dict(*args)
        # load the row of the table
        row_df = load_row(db, row_dict)
        files = self._load_files(*args)

        return row_df, files

    def _row_dict(self, *args: DataBase):
        return combine_metadata({}, *args)

    def _load_files(self, *args: DataBase):
        # return {"suffix1 (table col)": file1, "suffix2": file2}
        files = {}
        for data_instance in args:
            for suffix, _ in data_instance.save_to_file():
                # create unique filename: base_str + suffix
                filename = self.base_str + suffix
                # save the file
                file = np.load(self.file_folder + filename + ".npy")
                # add the filename to the filenames dict
                # --> will be added to the metadata table
                files[suffix] = file

        return files
