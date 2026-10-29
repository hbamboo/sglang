"""Copy the site launcher with separate shared experts on both PD roles."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
text = args.source.read_text()
for role in ("PREFILL", "DECODE"):
    anchor = f"  {role}_CMDS+=("
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.startswith(anchor)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {role} command anchor")
    index = matches[0]
    flag = "  cmd+=(--disable-shared-experts-fusion)\n"
    if index == 0 or lines[index - 1] != flag:
        lines.insert(index, flag)
    text = "".join(lines)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(text)
