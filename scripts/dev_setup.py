"""
Script to set up the development environment
"""

import os
import sys
import subprocess
import platform


def setup_virtualenv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    venv_dir = os.path.join(project_dir, "venv")

    if os.path.exists(venv_dir):
        print(f"Virtual environment already exists at {venv_dir}")
        return venv_dir

    print(f"Creating virtual environment at {venv_dir}")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    if platform.system() == "Windows":
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(venv_dir, "bin", "python")

    print("Installing development dependencies...")
    subprocess.check_call(
        [python_executable, "-m", "pip", "install", "--upgrade", "pip"]
    )
    subprocess.check_call(
        [python_executable, "-m", "pip", "install", "-e", project_dir + "[dev]"]
    )

    print("Development environment setup complete.")
    print(f"Activate the virtual environment with:")
    if platform.system() == "Windows":
        print(f"  {os.path.join(venv_dir, 'Scripts', 'activate')}")
    else:
        print(f"  source {os.path.join(venv_dir, 'bin', 'activate')}")

    return venv_dir


def run_tests(venv_dir):
    print("Running tests to verify setup...")

    if platform.system() == "Windows":
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(venv_dir, "bin", "python")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    result = subprocess.run(
        [python_executable, "-m", "pytest", os.path.join(project_dir, "tests")],
        capture_output=True,
        text=True,
    )

    print("Test output:")
    print(result.stdout)
    if result.stderr:
        print("Error output:")
        print(result.stderr)

    if result.returncode == 0:
        print("All tests passed! Setup is working correctly.")
    else:
        print("Some tests failed. Please check the output above.")


if __name__ == "__main__":
    venv_dir = setup_virtualenv()
    # run_tests(venv_dir)
