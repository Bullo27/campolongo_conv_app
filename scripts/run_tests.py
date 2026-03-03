#!/usr/bin/env python3
"""Run unit tests for the Conversation Timer app."""
import subprocess
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    env = {**os.environ,
           'ANDROID_HOME': os.path.expanduser('~/android-sdk'),
           'JAVA_HOME': os.path.expanduser('~/miniconda3'),
           'PATH': os.path.expanduser('~/miniconda3/bin') + ':' +
                   os.path.expanduser('~/android-sdk/cmdline-tools/latest/bin') + ':' +
                   os.path.expanduser('~/android-sdk/platform-tools') + ':' +
                   os.environ['PATH']}

    gradlew = PROJECT_ROOT / "gradlew"

    print("Running unit tests...")
    result = subprocess.run(
        [str(gradlew), "testDebugUnitTest"],
        cwd=str(PROJECT_ROOT),
        env=env
    )

    report = PROJECT_ROOT / "app" / "build" / "reports" / "tests" / "testDebugUnitTest" / "index.html"
    if report.exists():
        print(f"\nTest report: {report}")

    if result.returncode != 0:
        print("Tests failed!", file=sys.stderr)
        sys.exit(1)
    else:
        print("All tests passed!")

if __name__ == "__main__":
    main()
