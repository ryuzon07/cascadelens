"""The failure type of ``/api/v1``."""


class ApiFailure(Exception):
    """A failure with a code a client can act on."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
