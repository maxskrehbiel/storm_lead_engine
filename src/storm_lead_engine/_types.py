"""Array type aliases shared across modules."""

import numpy as np
import numpy.typing as npt

FloatGrid = npt.NDArray[np.float32]
FloatArray = npt.NDArray[np.float64]
BoolGrid = npt.NDArray[np.bool_]
CountGrid = npt.NDArray[np.int16]
RgbaImage = npt.NDArray[np.uint8]
