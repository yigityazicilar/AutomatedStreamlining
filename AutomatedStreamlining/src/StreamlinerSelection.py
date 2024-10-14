from enum import Enum

class StreamlinerSelection(Enum):
    COARSE = "coarse"
    FINE = "fine"
    FINE_FILTERED = "fine_filtered"

    def __str__(self):
        return self.value