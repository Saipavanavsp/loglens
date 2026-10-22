"""Download Loghub logs into data/raw/<source>/.

Sources (full logs from Zenodo record 8196385):
- linux    (25k lines)  syslog from a Linux server
- openssh  (655k lines) sshd on a public-facing host, lots of attack traffic
- apache   (56k lines)  Apache error log
- bgl      (4.7M lines) BlueGene/L supercomputer RAS log: kernel, memory and
                        hardware faults, with Loghub's alert flag per line

HDFS was considered but its public sample has only 16 distinct messages once
block ids and IPs are masked, so it adds little.

Only the single log file we need is read out of each archive; nothing else is
extracted to disk.

Usage: python -m loglens.data.download [--sample-only]
"""

import argparse
import hashlib
import io
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from loglens.data.paths import RAW

ZENODO = "https://zenodo.org/records/8196385/files/{name}?download=1"
GITHUB = "https://raw.githubusercontent.com/logpai/loghub/master/{path}"


@dataclass(frozen=True)
class Source:
    name: str
    log_file: str  # file name inside the archive
    archive: str  # Zenodo archive name
    sample_path: str  # 2k-line sample in the Loghub GitHub repo


SOURCES = [
    Source("linux", "Linux.log", "Linux.tar.gz", "Linux/Linux_2k.log"),
    Source("openssh", "SSH.log", "SSH.tar.gz", "OpenSSH/OpenSSH_2k.log"),
    Source("apache", "Apache.log", "Apache.tar.gz", "Apache/Apache_2k.log"),
    Source("bgl", "BGL.log", "BGL.zip", "BGL/BGL_2k.log"),
]


def _get(url: str) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=600) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _copy_member(archive_name: str, archive: bytes, wanted: str, dest: Path) -> None:
    """Stream one named file out of a .tar.gz or .zip into dest."""
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            name = next((n for n in zf.namelist() if Path(n).name == wanted), None)
            if name is None:
                raise FileNotFoundError(f"{wanted} not found in {archive_name}")
            with zf.open(name) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
        return
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == wanted), None)
        if member is None or not member.isfile():
            raise FileNotFoundError(f"{wanted} not found in {archive_name}")
        src = tar.extractfile(member)
        if src is None:
            raise FileNotFoundError(f"{wanted} in {archive_name} is not a regular file")
        with dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def download_source(src: Source, sample_only: bool = False, force: bool = False) -> Path:
    out_dir = RAW / src.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{src.name}.log"
    if dest.exists() and not force:
        print(f"[skip] {dest} exists")
        return dest
    if sample_only:
        print(f"[get ] {src.name}: {src.sample_path} (GitHub sample)")
        dest.write_bytes(_get(GITHUB.format(path=src.sample_path)))
    else:
        print(f"[get ] {src.name}: {src.archive} (Zenodo)")
        _copy_member(src.archive, _get(ZENODO.format(name=src.archive)), src.log_file, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    (out_dir / "SHA256").write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
    lines = sum(1 for _ in dest.open(encoding="utf-8", errors="replace"))
    print(f"[ok  ] {dest} ({lines:,} lines)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-only", action="store_true", help="use 2k-line samples only")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    for src in SOURCES:
        download_source(src, sample_only=args.sample_only, force=args.force)


if __name__ == "__main__":
    main()
