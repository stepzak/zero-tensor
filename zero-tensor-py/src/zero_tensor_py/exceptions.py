class ZeroTensorError(Exception):
    def __init__(self, message):
        super().__init__(message)


class MalformedMessageError(ZeroTensorError):
    def __init__(self, message):
        super().__init__(message)


class ProtocolError(ZeroTensorError):
    def __init__(self, message):
        super().__init__(message)


class ZTConnectionError(ZeroTensorError):
    def __init__(self, message):
        super().__init__(message)