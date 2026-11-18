"""Download backends: a local mirror and the Hugging Face Hub.

The rest of the launcher talks to a `Backend`, never to `huggingface_hub` directly. That is
what keeps the planner testable offline, and it is also why `--mirror <dir>` is not a test
stub: a lab that curates its own copy of the release uses the same code path as a laptop
pulling from the Hub.

`huggingface_hub` is imported inside the methods that need it. The package installs and
`symtrs get --list --manifest ...` works without it; only a download from the Hub needs the
`[hub]` extra, and the error says exactly that.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from symtrs.hub.errors import BackendError
from symtrs.hub.manifest import Repo, safe_relative_path

#: Where `symtrs get` looks when no --repo, --mirror or --manifest is given. Changing it is
#: a release decision, not a user preference.
DEFAULT_REPO = "safouaneelg/SyMTRS"
DEFAULT_REVISION = "main"
MANIFEST_NAME = "symtrs-hub.json"


class Backend:
    """The shape the installer and planner need. Subclasses implement the four methods."""

    repo: str = ""
    revision: str = ""

    def read_manifest_bytes(self) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def table(self, path: str) -> Iterator[Mapping[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def download(self, path: str, dest_root: Path) -> int:  # pragma: no cover
        raise NotImplementedError

    def pin(self, repo: str, revision: str) -> None:
        """Point subsequent file reads at the release the manifest declares."""
        self.repo = repo
        self.revision = revision

    def describe(self) -> str:
        return f"{self.repo}@{self.revision or '?'}"


class LocalBackend(Backend):
    """A directory that already contains the release: `--mirror <dir>`.

    The mirror is also what an offline machine and the tests use. Paths are resolved and
    checked against the mirror root again here, because a table is published data and a
    `..` in it must not reach outside the directory a user pointed at.
    """

    def __init__(self, root: Path, repo: str = "local mirror", revision: str = "local") -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise BackendError(
                f"mirror directory does not exist: {self.root}\n"
                f"  point --mirror at a checkout of the release (it must contain {MANIFEST_NAME})"
            )
        self.repo = repo
        self.revision = revision

    def _path(self, relative: str) -> Path:
        relative = safe_relative_path(relative, "mirror path")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            raise BackendError(
                f"mirror path escapes {self.root}: {relative!r}"
            ) from None
        return path

    def read_manifest_bytes(self) -> bytes:
        path = self._path(MANIFEST_NAME)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise BackendError(
                f"cannot read {path}: {exc}\n"
                f"  a mirror must contain {MANIFEST_NAME} at its root"
            ) from exc

    def table(self, path: str) -> Iterator[Mapping[str, Any]]:
        return _iter_jsonl(self._path(path))

    def download(self, path: str, dest_root: Path) -> int:
        source = self._path(path)
        if not source.is_file():
            raise BackendError(
                f"the mirror has no {path}\n"
                f"  run `symtrs get --list --mirror {self.root}` to see what it advertises"
            )
        destination = Path(dest_root) / safe_relative_path(path, "download path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return 0
        temporary = destination.with_name(destination.name + ".part")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        except OSError as exc:
            raise BackendError(f"cannot copy {source} to {destination}: {exc}") from exc
        return destination.stat().st_size

    def describe(self) -> str:
        return f"mirror {self.root}"


class HuggingFaceBackend(Backend):
    """The Hub. `repo_type="dataset"` for the release, `"model"` for a model vertical."""

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        *,
        repo_type: str = "dataset",
        revision: str = DEFAULT_REVISION,
        token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.repo = repo
        self.repo_type = repo_type
        self.revision = revision
        self.token = token
        self.cache_dir = None if cache_dir is None else str(cache_dir)

    @staticmethod
    def _hub():
        try:
            import huggingface_hub  # noqa: F401
        except ImportError as exc:
            raise BackendError(
                "downloading from the Hugging Face Hub needs huggingface_hub, which is not installed.\n"
                "  install it with:  pip install -e '.[hub]'\n"
                "  or work offline:  symtrs get --mirror <dir>"
            ) from exc
        return huggingface_hub

    def _errors(self):
        hub = self._hub()
        return hub.errors

    def _download_file(self, path: str, *, local_dir: Optional[Path] = None) -> Path:
        hub = self._hub()
        safe_relative_path(path, "repo path")
        try:
            return Path(
                hub.hf_hub_download(
                    repo_id=self.repo,
                    filename=path,
                    repo_type=self.repo_type,
                    revision=self.revision,
                    token=self.token,
                    cache_dir=self.cache_dir,
                    local_dir=None if local_dir is None else str(local_dir),
                )
            )
        except Exception as exc:  # translated below; never leak a library traceback for a 404
            raise self._translate(exc, path) from exc

    def _translate(self, exc: Exception, path: str) -> BackendError:
        errors = self._errors()
        gated = getattr(errors, "GatedRepoError", ())
        not_found = getattr(errors, "RepositoryNotFoundError", ())
        entry_missing = getattr(errors, "EntryNotFoundError", ())
        offline = getattr(errors, "OfflineModeIsEnabled", ())
        if offline and isinstance(exc, offline):
            return BackendError(
                f"offline mode is on and {self.repo}@{self.revision} is not cached.\n"
                "  use --mirror <dir>, or clear HF_HUB_OFFLINE to download"
            )
        if gated and isinstance(exc, gated):
            return BackendError(
                f"{self.repo} is gated: accept its terms at "
                f"https://huggingface.co/{self.repo_type}s/{self.repo} and authenticate "
                f"with `hf auth login` (or HF_TOKEN)."
            )
        if not_found and isinstance(exc, not_found):
            return BackendError(
                f"no {self.repo_type} repo {self.repo!r} at revision {self.revision!r}.\n"
                f"  check --repo/--revision, or run `symtrs get --list`"
            )
        if entry_missing and isinstance(exc, entry_missing):
            return BackendError(
                f"the release has no {path!r} at {self.repo}@{self.revision}.\n"
                f"  the manifest and the files disagree; re-fetch the manifest or use another revision"
            )
        return BackendError(
            f"cannot download {path!r} from {self.repo}@{self.revision}: {exc}"
        )

    def read_manifest_bytes(self) -> bytes:
        try:
            path = self._download_file(MANIFEST_NAME)
        except BackendError as exc:
            raise BackendError(
                f"{exc}\n"
                f"  a release this launcher can read carries {MANIFEST_NAME} at its root "
                f"(docs/hub.md). The pre-reformatted flat dataset does not: clone that one, "
                f"or wait for the reformatted release."
            ) from None
        return path.read_bytes()

    def table(self, path: str) -> Iterator[Mapping[str, Any]]:
        return _iter_jsonl(self._download_file(path))

    def download(self, path: str, dest_root: Path) -> int:
        destination = Path(dest_root) / safe_relative_path(path, "download path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Always ask the hub: with `local_dir` it compares etags and skips an intact file,
        # which is the resume behaviour we want, while a truncated file is replaced.
        self._download_file(path, local_dir=Path(dest_root))
        return destination.stat().st_size if destination.is_file() else 0

    def describe(self) -> str:
        return f"{self.repo}@{self.revision or '?'}"


class BackendSet:
    """One backend per repo the manifest declares, opened lazily and cached.

    This is the object the planner reads tables through and the installer downloads files
    through. It is what makes a multi-repo release ordinary: a group names its table repo,
    a file is routed to the view store or the metadata project, a checkpoint to its model
    vertical, and every read goes through the same factory.
    """

    def __init__(self, repos: Mapping[str, Repo], factory: Callable[[Repo], Backend]) -> None:
        self.repos = dict(repos)
        self.factory = factory
        self._backends: Dict[str, Backend] = {}

    def for_key(self, repo_key: str) -> Backend:
        backend = self._backends.get(repo_key)
        if backend is None:
            try:
                repo = self.repos[repo_key]
            except KeyError:
                raise BackendError(
                    f"no backend for repo {repo_key!r}; the manifest declares "
                    f"{', '.join(self.repos)}"
                ) from None
            backend = self.factory(repo)
            self._backends[repo_key] = backend
        return backend

    def table(self, repo_key: str, path: str) -> Iterator[Mapping[str, Any]]:
        return self.for_key(repo_key).table(path)

    def describe(self) -> str:
        return ", ".join(sorted(self.repos))


def mirror_backend_factory(root: Path) -> Callable[[Repo], Backend]:
    """Treat `root/<repo key>/` as one mirrored project.

    A local mirror is the same release, just on disk: the catalog in
    `<root>/metadata/symtrs-hub.json`, the images in `<root>/views/`, each vertical in its
    own directory. No repo has to be flattened or duplicated into a single tree.
    """
    root = Path(root)

    def make(repo: Repo) -> Backend:
        return LocalBackend(root / repo.key, repo=repo.repo, revision=repo.revision)

    return make


def huggingface_backend_factory(
    *, token: Optional[str] = None, cache_dir: Optional[Path] = None
) -> Callable[[Repo], Backend]:
    def make(repo: Repo) -> Backend:
        return HuggingFaceBackend(
            repo.repo,
            repo_type=repo.type,
            revision=repo.revision,
            token=token,
            cache_dir=cache_dir,
        )

    return make


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BackendError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
                if not isinstance(row, Mapping):
                    raise BackendError(f"{path}:{line_number} must be a JSON object")
                yield row
    except OSError as exc:
        raise BackendError(f"cannot read {path}: {exc}") from exc


__all__ = [
    "Backend",
    "BackendSet",
    "DEFAULT_REPO",
    "DEFAULT_REVISION",
    "HuggingFaceBackend",
    "LocalBackend",
    "MANIFEST_NAME",
    "huggingface_backend_factory",
    "mirror_backend_factory",
]
