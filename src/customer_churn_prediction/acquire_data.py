"""Download the immutable IBM Telco Customer Churn source file."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DEFAULT_OUTPUT = Path("data/raw/Telco-Customer-Churn.csv")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(output: Path = DEFAULT_OUTPUT) -> Path:
    """Download and verify the raw file without overwriting an existing source."""
    if output.exists():
        current_hash = sha256(output)
        if current_hash != EXPECTED_SHA256:
            raise RuntimeError(
                f"Raw file already exists with an unexpected SHA-256: {current_hash}"
            )
        print(f"Raw file already present and verified: {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    try:
        urllib.request.urlretrieve(SOURCE_URL, temporary)
        downloaded_hash = sha256(temporary)
        if downloaded_hash != EXPECTED_SHA256:
            raise RuntimeError(
                f"Downloaded file has an unexpected SHA-256: {downloaded_hash}"
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Downloaded and verified raw file: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    acquire(args.output)


if __name__ == "__main__":
    main()
