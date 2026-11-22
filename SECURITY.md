# Security

## Reporting a vulnerability

Do not put credentials, exploits, or private data into a public issue or PR.
Use GitHub's private vulnerability reporting on this repository if enabled.
If it is unavailable, open an issue requesting a private reporting channel
**without including vulnerability details**.

## Reviewing a repository made public

Public visibility includes reachable commits and other branches, not only the
current `main` tree. Removing a file in a new commit does not make old commits
private. CI logs, release bundles and previously downloaded clones must be
considered separately.

Review the complete Git history locally after fetching all refs:

```sh
git fetch --all --prune --tags
gitleaks git --log-opts='--all' --redact .
```

The [Secret History Scan](.github/workflows/secret-scan.yml) repeats a Gitleaks
scan on pull requests, pushes to `main`, weekly, and on demand. A clean scan
does not establish that no sensitive information has ever been published.

If a real credential is found, revoke or rotate it first. Then investigate
its exposure in all branches, Git history, Actions logs and releases.
Coordinate any history rewrite with downstream clones, forks, open PRs,
and existing tag references; do not rewrite history solely for tidiness.

The `.gitignore` protects against accidental future additions of common
local files and experiment outputs; it does not remove tracked files or
sanitize earlier commits.
