import numpy as np
from sqlalchemy import types


class NumpyArrayAdapter(types.TypeDecorator):
    impl = types.LargeBinary
    dtype: int = None
    count: int = None

    def process_bind_param(self, value, dialect):
        if value is not None:
            self.dtype = value.dtype
            self.count = len(value)
            return value.tobytes()

    def process_result_value(self, value, dialect):
        if value is not None:
            return np.frombuffer(value, dtype=self.dtype, count=self.count)
