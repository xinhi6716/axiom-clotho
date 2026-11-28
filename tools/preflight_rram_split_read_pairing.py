"""Compare independently labelled pre/post read rosters without reading outcomes."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import stat

FIELDS = ("chip", "address", "lane", "episode", "condition")
TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", re.ASCII)
INDEX = re.compile(r"[0-9]{1,18}", re.ASCII)
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 100_000


def roster(path):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("metadata input must be a regular file")
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("metadata input exceeds byte limit")
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    header = next(reader, [])
    if len(header) != len(FIELDS) or set(header) != set(FIELDS):
        raise ValueError("pre/post rosters require exactly chip,address,lane,episode,condition; omit read values")
    cols = [header.index(name) for name in FIELDS]
    result = {}
    for n, row in enumerate(reader, 1):
        if n > MAX_ROWS or len(row) != len(FIELDS):
            raise ValueError("row limit exceeded or wrong field count")
        chip, address, lane, episode, condition = (row[i] for i in cols)
        if not all(TOKEN.fullmatch(v) for v in (chip, episode, condition)):
            raise ValueError("invalid or missing declared chip, episode, or condition")
        if not INDEX.fullmatch(address) or not INDEX.fullmatch(lane):
            raise ValueError("address and lane require unsigned decimal indices")
        key = chip, int(address), int(lane), episode
        if key in result:
            raise ValueError("duplicate chip/address/lane/episode prevents one-to-one pairing")
        result[key] = condition
    if not result:
        raise ValueError("pre/post read roster is empty")
    return (info.st_dev, info.st_ino), result


def pair_support(pre, post):
    shared = pre.keys() & post.keys()
    conflicts = sum(pre[key] != post[key] for key in shared)
    return {
        "schema": "clotho.rram_split_read_pairing.v1",
        "pre_declared_rows": len(pre),
        "post_declared_rows": len(post),
        "pre_only_identities": len(pre.keys() - post.keys()),
        "post_only_identities": len(post.keys() - pre.keys()),
        "same_identity_conflicting_condition": conflicts,
        "exact_declared_identity_condition_pairs": len(shared) - conflicts,
        "all_declared_rows_pair_exactly": len(shared) == len(pre) == len(post) and not conflicts,
        "physical_pairing_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Each roster's identity must come from its own source acquisition metadata. "
        "Do not generate lane labels from row order or copy one roster for both reads. "
        "Matching declarations do not verify physical lineage or elapsed time."
    ))
    parser.add_argument("--pre-roster", type=Path, required=True)
    parser.add_argument("--post-roster", type=Path, required=True)
    args = parser.parse_args()
    try:
        pre_file, pre = roster(args.pre_roster)
        post_file, post = roster(args.post_roster)
        if pre_file == post_file:
            raise ValueError("pre and post rosters must be distinct inputs")
    except (OSError, UnicodeError, csv.Error):
        parser.exit(2, "error: could not read UTF-8 roster CSV\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    report = pair_support(pre, post)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(not report["all_declared_rows_pair_exactly"])


if __name__ == "__main__":
    raise SystemExit(main())
