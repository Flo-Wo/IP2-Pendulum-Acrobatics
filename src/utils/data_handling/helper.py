import logging

import pandas as pd

from utils import DataBase


def generate_base_str(*args: DataBase) -> str:
    # we just add the string representation of all of them
    base_str = ""
    for data_instance in args:
        data_str = str(data_instance)
        if data_str != "":
            base_str += data_str + "_"
    # NOTE: we keep the last "_" to simply add the suffix strings of the
    # save_to_file dict entries
    return base_str


def combine_metadata(metadata: dict, *args: DataBase) -> dict:
    for data_instance in args:
        metadata.update(data_instance.save_to_metadata())
    # add_metadata holds all new metadata --> to easily extend the
    # current metadata table, we sort the new data
    return dict(sorted(metadata.items()))


def combine_indices(metadata: dict, *args: DataBase) -> dict:
    """Combine dict of file suffixes and filenames with metadata dict.

    We ignore parameters of the "ignore" field to built a unique index of
    the database.

    Parameters
    ----------
    metadata : dict
        Dict of the form: suffix + filename for each file to be saved.

    Returns
    -------
    dict
        Keys are combined the unique key for the database.
    """
    for data_instance in args:
        metadata.update(data_instance.filter_values())
    # add_metadata holds all new metadata --> to easily extend the
    # current metadata table, we sort the new data
    return dict(sorted(metadata.items()))


def already_computed(df: pd.DataFrame, row: dict) -> bool:
    """Check whether row already exists inside the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Database to be checked.
    row : dict
        Row which is either included or not.

    Returns
    -------
    bool
        True, if row is contained, False, if not.
    """
    if df is None:
        return False
    return not (load_row(df, row).empty)


def load_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    # source:
    # https://stackoverflow.com/questions/34157811/filter-a-pandas-dataframe-using-values-from-a-dict
    # return df.loc[(df[list(row)] == pd.DataFrame.from_dict([row])).all(axis=1)]
    return df.loc[(df[list(row)] == pd.Series([row])).all(axis=1)]


# helper functions to load and save the pandas metadata table
def read_db(path_metadata: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path_metadata)
    except:
        logging.info("Database not found!")
        return None


def save_db(path_metadata: str, db: pd.DataFrame) -> None:
    db.to_csv(path_metadata, index=False)
