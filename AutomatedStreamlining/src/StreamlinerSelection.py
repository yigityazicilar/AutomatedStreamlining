from enum import Enum


class StreamlinerSelection(Enum):
    COARSE = "coarse"
    FINE = "fine"
    FINE_FILTERED = "fine_filtered"
    ALL = "all"

    def __str__(self):
        return self.value
