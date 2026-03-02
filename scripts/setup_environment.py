#!/usr/bin/env python3
"""
Setup Android SDK and gh CLI for the Conversation Timer project.
Downloads and installs everything headlessly — no Android Studio needed.
"""

import os
import subprocess
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

ANDROID_HOME = Path.home() / "android-sdk"
CMDLINE_TOOLS_URL = "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
CMDLINE_TOOLS_DIR = ANDROID_HOME / "cmdline-tools" / "latest"


def run(cmd, check=True, env=None, input_text=None):
    """Run a command, printing it first."""
    print(f"  >>> {cmd}")
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(
        cmd, shell=True, check=check, env=merged_env,
        capture_output=True, text=True, input=input_text
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0 and result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return result


def setup_android_sdk():
    """Download and install Android SDK command-line tools."""
    print("\n=== Setting up Android SDK ===")

    if CMDLINE_TOOLS_DIR.exists() and (CMDLINE_TOOLS_DIR / "bin" / "sdkmanager").exists():
        print("Android SDK command-line tools already installed, skipping download.")
    else:
        # Download
        zip_path = Path("/tmp/android-cmdline-tools.zip")
        if not zip_path.exists():
            print(f"Downloading command-line tools from {CMDLINE_TOOLS_URL}...")
            urllib.request.urlretrieve(CMDLINE_TOOLS_URL, zip_path)
            print("Download complete.")
        else:
            print("Using cached download.")

        # Extract
        extract_dir = Path("/tmp/android-cmdline-tools-extract")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

        # Move to proper location
        CMDLINE_TOOLS_DIR.parent.mkdir(parents=True, exist_ok=True)
        if CMDLINE_TOOLS_DIR.exists():
            shutil.rmtree(CMDLINE_TOOLS_DIR)
        shutil.move(str(extract_dir / "cmdline-tools"), str(CMDLINE_TOOLS_DIR))
        # Fix permissions on extracted binaries
        for bin_file in (CMDLINE_TOOLS_DIR / "bin").iterdir():
            bin_file.chmod(0o755)
        print(f"Installed to {CMDLINE_TOOLS_DIR}")

    # Set environment variables
    sdk_env = {
        "ANDROID_HOME": str(ANDROID_HOME),
        "PATH": f"{CMDLINE_TOOLS_DIR / 'bin'}:{ANDROID_HOME / 'platform-tools'}:{os.environ['PATH']}"
    }

    # Accept licenses
    print("\nAccepting SDK licenses...")
    run(
        f"{CMDLINE_TOOLS_DIR / 'bin' / 'sdkmanager'} --sdk_root={ANDROID_HOME} --licenses",
        check=False, env=sdk_env,
        input_text="\n".join(["y"] * 20)
    )

    # Install SDK components
    components = [
        "platform-tools",
        "platforms;android-35",
        "build-tools;35.0.0",
    ]
    print("\nInstalling SDK components...")
    for component in components:
        print(f"\n  Installing {component}...")
        run(
            f"{CMDLINE_TOOLS_DIR / 'bin' / 'sdkmanager'} --sdk_root={ANDROID_HOME} \"{component}\"",
            env=sdk_env
        )

    # Update shell profile
    bashrc = Path.home() / ".bashrc"
    marker = "# Android SDK (Conversation Timer setup)"
    bashrc_content = bashrc.read_text() if bashrc.exists() else ""
    if marker not in bashrc_content:
        print("\nAdding Android SDK to ~/.bashrc...")
        with open(bashrc, "a") as f:
            f.write(f"\n{marker}\n")
            f.write(f'export ANDROID_HOME="{ANDROID_HOME}"\n')
            f.write(f'export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"\n')
    else:
        print("\n~/.bashrc already has Android SDK paths.")

    # Verify
    print("\nVerifying installation...")
    run(f"{CMDLINE_TOOLS_DIR / 'bin' / 'sdkmanager'} --sdk_root={ANDROID_HOME} --list | head -20", env=sdk_env)
    print("\nAndroid SDK setup complete!")
    return sdk_env


def setup_gh_cli():
    """Install GitHub CLI."""
    print("\n=== Setting up GitHub CLI ===")

    # Check if already installed
    result = run("gh --version", check=False)
    if result.returncode == 0:
        print("gh CLI already installed.")
        return

    # Try conda first (user has miniconda3)
    print("Installing gh via conda-forge...")
    result = run("conda install -y -c conda-forge gh", check=False)
    if result.returncode == 0:
        print("gh CLI installed successfully via conda.")
        return

    # Fallback: install via apt
    print("conda install failed, trying apt...")
    run("sudo mkdir -p /etc/apt/keyrings", check=False)
    run('curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null', check=False)
    run('echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null', check=False)
    run("sudo apt update && sudo apt install -y gh", check=False)

    # Verify
    result = run("gh --version", check=False)
    if result.returncode == 0:
        print("gh CLI installed successfully.")
    else:
        print("WARNING: Could not install gh CLI. You can install it manually later.")


def verify_java():
    """Verify Java is available."""
    print("\n=== Verifying Java ===")
    result = run("java -version", check=False)
    if result.returncode != 0:
        print("ERROR: Java not found. Please install JDK 17+.")
        sys.exit(1)
    print("Java is available.")


def main():
    print("=" * 60)
    print("  Conversation Timer — Environment Setup")
    print("=" * 60)

    verify_java()
    sdk_env = setup_android_sdk()
    setup_gh_cli()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print(f"\n  ANDROID_HOME = {ANDROID_HOME}")
    print(f"  Run 'source ~/.bashrc' to update your current shell.")
    print(f"  Then run 'gh auth login' to authenticate with GitHub.")


if __name__ == "__main__":
    main()
