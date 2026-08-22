import os
import tarfile
import tempfile
from pathlib import PurePosixPath


def unpack_model(model_path: str) -> tuple[str, bool]:
    if os.path.isdir(model_path):
        return model_path, False
    with tarfile.open(model_path, "r") as tar:
        tmp_dir = tempfile.mkdtemp(prefix="fnnx_")
        tar.extractall(tmp_dir, members=_extractable_members(tar), filter="data")
    return tmp_dir, True


def _extractable_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Members that may reach the disk: regular files and directories at safe paths.

    Symbolic links, hard links, device nodes, absolute paths and `..` segments are all
    ignored rather than written out.
    """
    return [
        member
        for member in tar.getmembers()
        if (member.isfile() or member.isdir()) and not _is_unsafe_path(member.name)
    ]


def _is_unsafe_path(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or name.startswith("\\")
    )
