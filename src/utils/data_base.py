from abc import ABC, abstractmethod
from typing import List


class DataBase(ABC):
    @abstractmethod
    def __str__(self):
        """Generate a unique string used to save data to files."""
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self) -> dict:
        """Part of the attributes which should be saved to files.
        The dict is of the form
        {"prefix1" : data1, "prefix2": data2}
        """
        raise NotImplementedError

    @abstractmethod
    def save_to_metadata(self) -> dict:
        """Part of the attributes which will be saved to the metadata table.
        The dict is of the form
        {"col1" : data1, "col2": data2}
        """
        raise NotImplementedError

    # does not have to be an abstractmethod, as a default version is provided
    def exclude_index(self) -> List[str]:
        """Parts of the attributes which won't be used for the metatable index.
        Per default we return an empty list, but one can exlcude some parts of the
        metadata dict (e.g. the computational time, as the experiment is the same,
        but the time changes).
        """
        return []

    def filter_values(self) -> dict:
        """Return only parts of the metadata dict, which should be used to search for
        a row. I.e. we take the metadata dict and exclude parts of the index.
        """
        return {
            key: value
            for key, value in self.save_to_metadata().items()
            if key not in self.exclude_index()
        }
