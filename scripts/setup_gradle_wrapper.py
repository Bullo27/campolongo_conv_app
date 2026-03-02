#!/usr/bin/env python3
"""Download and set up Gradle wrapper for the project."""
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    env = {**os.environ,
           'ANDROID_HOME': os.path.expanduser('~/android-sdk'),
           'PATH': os.path.expanduser('~/android-sdk/cmdline-tools/latest/bin') + ':' +
                   os.path.expanduser('~/android-sdk/platform-tools') + ':' +
                   os.environ['PATH']}

    # Check if gradle is available
    result = subprocess.run(['which', 'gradle'], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        # Download gradle wrapper using a minimal Gradle
        print("Downloading Gradle wrapper...")
        import urllib.request
        import zipfile
        import shutil

        gradle_url = "https://services.gradle.org/distributions/gradle-8.11.1-bin.zip"
        zip_path = Path("/tmp/gradle-bin.zip")
        extract_dir = Path("/tmp/gradle-extract")

        if not (extract_dir / "gradle-8.11.1").exists():
            if not zip_path.exists():
                print(f"Downloading Gradle from {gradle_url}...")
                urllib.request.urlretrieve(gradle_url, zip_path)
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)

        gradle_bin = str(extract_dir / "gradle-8.11.1" / "bin" / "gradle")
        os.chmod(gradle_bin, 0o755)
    else:
        gradle_bin = result.stdout.strip()

    # Generate wrapper
    print("Generating Gradle wrapper...")
    subprocess.run(
        [gradle_bin, 'wrapper', '--gradle-version', '8.11.1'],
        cwd=str(PROJECT_ROOT),
        check=True,
        env=env
    )

    # Make gradlew executable
    gradlew = PROJECT_ROOT / "gradlew"
    os.chmod(str(gradlew), 0o755)
    print(f"Gradle wrapper created at {gradlew}")

if __name__ == "__main__":
    main()
