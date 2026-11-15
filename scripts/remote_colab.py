from __future__ import annotations

import os
import subprocess


def main() -> None:
    command = os.environ.get("REMOTE_CMD")
    if not command:
        raise ValueError("REMOTE_CMD is required")
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd="/content/namellm",
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    print(f"remote_exit_code={result.returncode}")


if __name__ == "__main__":
    main()
