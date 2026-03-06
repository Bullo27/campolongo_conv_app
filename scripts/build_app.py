#!/usr/bin/env python3
"""Build the Conversation Timer APK."""
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

    build_type = sys.argv[1] if len(sys.argv) > 1 else "assembleDebug"
    gradlew = PROJECT_ROOT / "gradlew"

    print(f"Building {build_type}...")
    result = subprocess.run(
        [str(gradlew), build_type],
        cwd=str(PROJECT_ROOT),
        env=env
    )

    if result.returncode == 0:
        # Determine output directory based on build type
        if "Release" in build_type or "release" in build_type:
            variant = "release"
        else:
            variant = "debug"

        apk_dir = PROJECT_ROOT / "app" / "build" / "outputs" / "apk" / variant
        if apk_dir.exists():
            apks = list(apk_dir.glob("*.apk"))
            if apks:
                apk = apks[0]
                size_mb = apk.stat().st_size / (1024 * 1024)
                print(f"\nAPK built: {apk}")
                print(f"APK size: {size_mb:.1f} MB")
    else:
        print("Build failed!", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
