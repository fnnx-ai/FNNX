class CompileError(Exception):
    """Raised when a model cannot be compiled to C."""


class HarnessError(Exception):
    """Raised when a compiled artifact cannot be built, loaded, or driven from Python."""
