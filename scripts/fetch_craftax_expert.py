"""Fetch the published PPO-RNN checkpoint for Craftax-Classic."""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

REPO = "mathisweil/remdm-craftax-checkpoints"
REVISION = "5b66c77b497777e4ffdcbf20c3563402e404daf8"
SOURCE = "checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M/"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_CHECKPOINT = RESULTS_DIR / "checkpoints/craftax-classic-ppo-rnn"


def fetch(out: Path) -> None:
    if out.exists():
        marker = out / "source.json"
        if marker.is_file() and json.loads(marker.read_text())["revision"] == REVISION:
            print(out)
            return
        raise FileExistsError(f"Choose a new checkpoint output directory: {out}")

    api = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"
    with urlopen(api, timeout=20) as stream:
        files = [
            row["rfilename"]
            for row in json.load(stream)["siblings"]
            if row["rfilename"].startswith(SOURCE)
        ]
    if not files:
        raise ValueError(f"No checkpoint files found at revision {REVISION}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".craftax-expert-", dir=out.parent) as temporary:
        staging = Path(temporary) / "checkpoint"

        def download(name: str) -> None:
            relative = Path(name.removeprefix(SOURCE))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid checkpoint path: {name}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
            with urlopen(url, timeout=30) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)

        with ThreadPoolExecutor(max_workers=4) as executor:
            for _ in executor.map(download, files):
                pass
        (staging / "source.json").write_text(
            json.dumps({"repo": REPO, "revision": REVISION, "path": SOURCE}) + "\n"
        )
        staging.rename(out)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    fetch(parser.parse_args().out)
