import logging
import re
import sys
from os import path
from subprocess import check_output

import pandas as pd

commit_re = re.compile(
    r"""
^(?P<author>.*?)
\s
(?P<timestamp>\d+-\d+-\d+\s\d+:\d+:\d+\s-\d+)
\s
(?P<message>.*)$
""",
    re.VERBOSE,
)


def get_commits(repo_path: str) -> str:
    log_format = "%n###%n%ae %ai %s"
    command = ["git", "-C", repo_path, "log", "--stat", f"--pretty={log_format}"]
    logging.debug("[wrangle.py#get_commits] command=" + " ".join(command))
    return check_output(command).decode("utf8").strip()


def handle_section(text):
    lines = text.split("\n")
    data = re.match(commit_re, lines[0]).groupdict()

    if len(lines) == 1:
        return data

    lines = "\n".join(lines[1:]).strip().split("\n")
    files_changed = lines[-1].strip()

    data["n_files_changed"] = int(re.search(r"^\d+", files_changed[0])[0])

    if "insertions" in files_changed:
        data["insertions"] = int(re.search("(\d+)\sinsertions", files_changed)[1])
    if "deletions" in files_changed:
        data["deletions"] = int(re.search("(\d+)\sdeletions", files_changed)[1])

    return data


def get_commit_df(repo_path, use_cache=True) -> pd.DataFrame:
    contents = get_commits(repo_path)
    sections = [
        section.strip() for section in contents.split("###\n") if section.strip()
    ]
    df = pd.DataFrame([handle_section(section) for section in sections])

    for col in "n_files_changed", "insertions", "deletions":
        df[col] = df[col].fillna(0).astype(int)

    df.timestamp = pd.to_datetime(df.timestamp, utc=True)
    df = df.set_index("timestamp").tz_convert("America/Chicago").sort_index()

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.prog = "python -m zgulde.gitlog"
    parser.description = "Export git log information as a csv"
    parser.add_argument(
        "--repo-path",
        help="Local path to the git repo to be analyzed (default this directory)",
        default=".",
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="count"
    )
    args = parser.parse_args()

    if args.verbose is None:
        loglevel = logging.WARN
    elif args.verbose == 1:
        loglevel = logging.INFO
    elif args.verbose >= 2:
        loglevel = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=loglevel,
    )

    df = get_commit_df(args.repo_path)
    print(df.to_csv(index=True))
