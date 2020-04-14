import argparse
import json
import sys

from .gitlog import get_commits

parser = argparse.ArgumentParser()
parser.prog = "python -m zgulde.gitlog"
parser.description = "Export git log information in JSON format"
parser.add_argument(
    "--repo-path",
    default=".",
    help="path to the repo to be analyzed (default this directory)",
)
parser.add_argument(
    "-o",
    "--outfile",
    type=argparse.FileType("w"),
    default=sys.stdout,
    help="Where to output the processed json data (default stdout)",
)
parser.add_argument(
    "-i",
    "--indent",
    default=2,
    type=int,
    help="Indentation of the resulting json (default: %(default)s)",
)
args = parser.parse_args()

commits = get_commits(args.repo_path)
json.dump(commits, args.outfile, indent=args.indent)
