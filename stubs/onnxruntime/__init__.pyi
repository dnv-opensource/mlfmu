import os
from collections.abc import Sequence
from typing import Any

class Dummy: ...

class NodeArg:
    name: str
    shape: list[int]

class Session: ...
class SessionOptions: ...

class InferenceSession(Session):
    def __init__(
        self,
        path_or_bytes: str | bytes | os.PathLike[str],
        sess_options: SessionOptions | None = None,
        providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def get_outputs(self) -> list[NodeArg]: ...
