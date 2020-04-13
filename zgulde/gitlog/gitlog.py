import git
from tqdm import tqdm


def to_dict(commit):
    subject, *body = commit.message.split("\n")
    body = "\n".join(body).strip()
    return {
        "author_email": commit.author.email,
        "author_name": commit.author.name,
        "authored_at": str(commit.authored_datetime),
        "body": body,
        "committed_at": str(commit.committed_datetime),
        "committer_email": commit.committer.email,
        "committer_name": commit.committer.name,
        "deletions": commit.stats.total["deletions"],
        "files_changed": commit.stats.total["files"],
        "insertions": commit.stats.total["insertions"],
        "lines_changed": commit.stats.total["lines"],
        "message": commit.message,
        "n_parents": len(commit.parents),
        "parents": [parent.hexsha for parent in commit.parents],
        "sha": commit.hexsha,
        "size": commit.size,
        "subject": subject,
        "type": commit.type,
    }


def get_commits(repo_path):
    repo = git.Repo(repo_path)
    commits = [to_dict(commit) for commit in tqdm(list(repo.iter_commits()))]
    return commits


def get_commits_df(repo_path):
    import pandas as pd

    df = pd.DataFrame(get_commits(repo_path))
    df.committed_at = pd.to_datetime(df.committed_at, utc=True)
    df.authored_at = pd.to_datetime(df.authored_at, utc=True)
    return df
