import hashlib
import numpy as np
from functools import lru_cache, wraps

# https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays/76483281#76483281
class YetAnotherWrapper:
    def __init__(self, x: np.array) -> None:
        self.values = x
        # here you can use your own hashing function
        self.h = hashlib.sha224(x.view()).hexdigest()

    def __hash__(self) -> int:
        return hash(self.h)

    def __eq__(self, __value: object) -> bool:
        return __value.h == self.h

def memoize(expensive_function):
    @lru_cache()
    def cached_wrapper(shell):
        return expensive_function(shell.values)

    @wraps(expensive_function)
    def wrapper(x: np.array):
        shell = YetAnotherWrapper(x)
        return cached_wrapper(shell)

    return wrapper