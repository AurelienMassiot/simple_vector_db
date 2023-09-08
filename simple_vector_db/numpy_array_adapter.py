import numpy as np
from sqlalchemy import types


class NumpyArrayAdapter(types.TypeDecorator):
    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        if value is not None:
            return value.tobytes()

    def process_result_value(self, value, dialect):
        if value is not None:
            return np.frombuffer(value, dtype=np.float64)
