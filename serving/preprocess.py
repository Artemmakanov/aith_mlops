from typing import Any

import numpy as np


class Preprocess(object):
    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        return body["x"]

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> list:
        return data.tolist() if isinstance(data, np.ndarray) else list(data)
