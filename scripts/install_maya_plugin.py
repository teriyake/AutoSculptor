import os
import sys
import shutil
from pathlib import Path


def install_plugin():
    is_windows = sys.platform.startswith("win")
    is_mac = sys.platform == "darwin"

    if is_windows:
        maya_dir = os.path.join(os.environ["USERPROFILE"], "Documents", "maya")
    elif is_mac:
        maya_dir = os.path.join(
            os.environ["HOME"], "Library", "Preferences", "Autodesk", "maya"
        )

    maya_versions = [d for d in os.listdir(maya_dir) if d.isdigit()]

    for version in maya_versions:
        plugin_dir = os.path.join(maya_dir, version, "plug-ins")
        scripts_dir = os.path.join(maya_dir, version, "scripts")

        os.makedirs(plugin_dir, exist_ok=True)
        os.makedirs(scripts_dir, exist_ok=True)

        source_plugin_file = (
            Path(__file__).parent.parent / "autosculptor" / "maya" / "plugin.py"
        )
        source_package_dir = Path(__file__).parent.parent / "autosculptor"
        destination_package_dir = Path(scripts_dir) / "autosculptor"

        shutil.copy(
            source_plugin_file, os.path.join(plugin_dir, "autosculptor_plugin.py")
        )
        try:
            if destination_package_dir.exists() and destination_package_dir.is_dir():
                shutil.rmtree(destination_package_dir)
            shutil.copytree(source_package_dir, destination_package_dir)
            print(f"Installed AutoSculptor plugin and package for Maya {version}")
            print(
                f"  - Plugin file: {os.path.join(plugin_dir, 'autosculptor_plugin.py')}"
            )
            print(f"  - Package directory: {destination_package_dir}")
        except Exception as e:
            print(f"Error installing AutoSculptor package for Maya {version}: {e}")
            print(
                f"  - Plugin file (only) might be installed at: {os.path.join(plugin_dir, 'autosculptor_plugin.py')}"
            )


if __name__ == "__main__":
    install_plugin()
