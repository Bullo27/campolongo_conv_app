#!/usr/bin/env python3
"""Initialize git repo and push ConversationTimer project to GitHub."""

import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/Bullo27/campolongo_conv_app.git"

GITIGNORE_CONTENT = """\
# Build outputs
build/
app/build/
.gradle/

# Local config
local.properties

# IDE
.idea/
*.iml

# OS
.DS_Store
Thumbs.db

# APK files (build outputs)
*.apk
*.aab
"""

def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"STDERR: {result.stderr.strip()}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result

def main():
    os.chdir(PROJECT_ROOT)
    print(f"Working in: {PROJECT_ROOT}")

    # Write .gitignore
    gitignore_path = PROJECT_ROOT / ".gitignore"
    gitignore_path.write_text(GITIGNORE_CONTENT)
    print("Created .gitignore")

    # Initialize git repo
    if not (PROJECT_ROOT / ".git").exists():
        run(["git", "init"])
        run(["git", "branch", "-M", "main"])
    else:
        print("Git repo already initialized")

    # Configure git user (local to this repo only)
    run(["git", "config", "user.email", "masterbullo@gmail.com"])
    run(["git", "config", "user.name", "Bullo27"])

    # Add remote
    result = subprocess.run(["git", "remote", "get-url", "origin"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        run(["git", "remote", "add", "origin", REPO_URL])
    else:
        print(f"Remote origin already set: {result.stdout.strip()}")

    # Stage all files
    run(["git", "add", "-A"])

    # Show what will be committed
    run(["git", "status"])

    # Commit
    run(["git", "commit", "-m", "Initial commit: Conversation Timer v1\n\n"
         "Native Android app (Kotlin + Jetpack Compose) with on-device speaker\n"
         "diarization via MFCC + Silero VAD. Tracks 11 conversation timing\n"
         "metrics between 2 speakers in real-time.\n\n"
         "Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"])

    # Force push (to replace the auto-init README)
    # Using token from environment or git credential
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        push_url = f"https://{token}@github.com/Bullo27/campolongo_conv_app.git"
        run(["git", "remote", "set-url", "origin", push_url])

    run(["git", "push", "-u", "origin", "main", "--force"])
    print("\nSuccessfully pushed to GitHub!")
    print(f"Repo: {REPO_URL}")

if __name__ == "__main__":
    main()
