"""Extract and validate Lego from the official NeRF example archive."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from PIL import Image

root = Path(__file__).resolve().parents[1]
archive = root / "data/nerf_example_data.zip"
destination = root / "data"
with ZipFile(archive) as z:
    members = [n for n in z.namelist() if n.startswith("nerf_synthetic/lego/")]
    assert members, "Archive has no Lego scene"
    for name in members:
        target = (destination / name).resolve()
        assert target.is_relative_to(destination.resolve())
        z.extract(name, destination)
scene = destination / "nerf_synthetic/lego"
report = {"source": "https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip",
          "archive_sha256": hashlib.file_digest(archive.open("rb"), "sha256").hexdigest(), "splits": {}}
for split in ("train", "val", "test"):
    meta = json.loads((scene / f"transforms_{split}.json").read_text())
    sizes = set()
    for frame in meta["frames"]:
        with Image.open(scene / (frame["file_path"] + ".png")) as im:
            sizes.add(im.size)
            im.verify()
    report["splits"][split] = {"frames": len(meta["frames"]), "sizes": sorted(sizes)}
(destination / "lego_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
