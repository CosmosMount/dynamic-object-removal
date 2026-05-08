from __future__ import annotations

from pathlib import Path

from object_removal.io.masks import read_mask_u8, to_binary_255, write_mask_u8, list_mask_files


def run(*, in_masks_dir: Path, out_masks_dir: Path) -> dict:
    files = list_mask_files(in_masks_dir)
    if not files:
        raise ValueError(f"No masks found under: {in_masks_dir}")
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        m = read_mask_u8(p)
        write_mask_u8(out_masks_dir / p.name, to_binary_255(m))
    return {"num_frames": len(files)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Identity tracker: copy masks to canonical binary format")
    ap.add_argument("--in_masks_dir", required=True)
    ap.add_argument("--out_masks_dir", required=True)
    args = ap.parse_args()
    run(in_masks_dir=Path(args.in_masks_dir), out_masks_dir=Path(args.out_masks_dir))


if __name__ == "__main__":
    main()

