from utils.data_base import DataBase
from typing import List


class AddData(DataBase):
    def __init__(
        self,
        to_metadata: dict = {},
        to_files: dict = {},
        str_repr: str = "",
        exclude: List[str] = [],
    ):
        self.metadata = to_metadata
        self.files = to_files
        self.str_repr = str_repr
        self.exclude = exclude

    def __str__(self):
        return self.str_repr

    def save_to_file(self) -> dict:
        return self.files

    def save_to_metadata(self) -> dict:
        return self.metadata

    def exclude_index(self) -> List[str]:
        return self.exclude
