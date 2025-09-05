from src.logging import log_error


class ShapeMismatchedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        log_error(message)

class SingularError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        log_error(message)

class MaxIteratonError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        log_error(message)

