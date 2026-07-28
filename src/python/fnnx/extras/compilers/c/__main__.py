"""Command-line entry: `python -m fnnx.extras.compilers.c <source> -o <dir>`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from fnnx.extras.compilers.c.errors import CompileError

if TYPE_CHECKING:
    from fnnx.extras.compilers.c.result import CompileResult

ONNX_SUFFIX = ".onnx"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = _compile(arguments)
    except (CompileError, OSError) as error:
        # OSError covers an unusable `-o`: a path that is a file, or one nothing may
        # write to. Nothing the compiler itself raises reaches here as an OSError.
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("\n".join(_summary(arguments.source, result)))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fnnx.extras.compilers.c",
        description=(
            "Compile an FNNX pipeline bundle or an ONNX model into a single "
            "self-contained C99 header."
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        help="FNNX bundle (directory or tar) or `.onnx` model file to compile",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="directory the header and the compile report are written to",
    )
    parser.add_argument(
        "--dim",
        action="append",
        default=[],
        dest="dims",
        type=_dim_binding,
        metavar="NAME=VALUE",
        help=(
            "bind a symbolic dimension to a size; repeatable. "
            "Dimensions left unbound default to 1"
        ),
    )
    parser.add_argument(
        "--runtime-dim",
        action="append",
        default=[],
        dest="runtime_dims",
        type=_dim_binding,
        metavar="NAME=MAX",
        help=(
            "leave a symbolic dimension to be sized per call, up to MAX; repeatable. "
            "Buffers are sized for MAX and every entrypoint takes the actual value"
        ),
    )
    parser.add_argument(
        "--prefix",
        help=(
            "prefix for the emitted files and every public symbol "
            "(default: the model's own name)"
        ),
    )
    return parser


def _dim_binding(text: str) -> tuple[str, int]:
    name, separator, size = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got `{text}`")
    try:
        return name, int(size)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"dimension `{name}` needs an integer size, got `{size}`"
        ) from None


def _compile(arguments: argparse.Namespace) -> CompileResult:
    return _entrypoint(arguments.source)(
        arguments.source,
        arguments.output_dir,
        dim_bindings=dict(arguments.dims),
        runtime_dims=dict(arguments.runtime_dims),
        prefix=arguments.prefix,
    )


def _entrypoint(source: Path) -> Callable[..., CompileResult]:
    """The compiler the source asks for: `.onnx` is a model, anything else a bundle."""
    try:
        from fnnx.extras.compilers.c import compile_bundle, compile_onnx
    except ModuleNotFoundError as error:
        # The optional `onnx` dependency, missing; its own message names what to install.
        raise CompileError(str(error)) from error
    return compile_onnx if source.suffix == ONNX_SUFFIX else compile_bundle


def _summary(source: Path, result: CompileResult) -> list[str]:
    report = result.report
    memory = report["memory"]
    fields = [
        ("header", str(result.header_path)),
        ("report", str(result.report_path)),
        ("entrypoint", f"{report['entrypoint']['symbol']}()"),
        ("opsets", _pairs(report["opsets"])),
        ("dimensions", _pairs(report["dim_bindings"]) or "none"),
        (
            "runtime dimensions",
            ", ".join(f"{dim['name']}<={dim['max']}" for dim in report["runtime_dims"])
            or "none",
        ),
        ("kernels", str(len(report["kernels"]))),
        (
            "static memory",
            f"{memory['static_bytes']} bytes (weights {memory['weights_bytes']}, "
            f"arena {memory['arena_bytes']})",
        ),
    ]
    width = max(len(label) for label, _ in fields) + 2
    return [f"Compiled `{source}`:"] + [
        f"  {label + ':':<{width}}{value}" for label, value in fields
    ]


def _pairs(mapping: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in mapping.items())


if __name__ == "__main__":
    raise SystemExit(main())
