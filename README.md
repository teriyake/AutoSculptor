# AutoSculptor

## Installation

### Requirements

- Python 3.10.8 or later (must match the Python version used by your Maya installation)
- NumPy
- PyQt5
- Maya 2024 or later

### Development Setup

1. Clone the repository:

```bash
git clone https://github.com/your-org/autosculptor.git autosculptor && cd autosculptor
```

2. Set up the development environment:

```bash
# For Windows
python scripts/dev_setup.py

# For macOS/Linux
python3 scripts/dev_setup.py
```

3. Activate the virtual environment:

```bash
# For Windows
venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate
```

4. Install for development:

```bash
pip install -e .
```

### Maya Plugin Installation

To install the plugin for Maya:

```bash
# For standard installation
python scripts/install_maya_plugin.py

# For development mode (creates symlinks)
python scripts/install_maya_plugin.py --dev
```

Once installed, you can load the plugin in Maya using Plugin Manager or in Script Editor with:

```python
import maya.cmds as cmds
cmds.loadPlugin('autosculptor.py')
```

## Functionalities Implemented

### Basic Brush Class & Mesh Interaction

To test basic `Brush` functionalities and Maya mesh interaction:

1. Load the plugin
2. Open the Script Editor and run `test_ui.py`

### Stroke Recording

To test stroke recording: 

1. Load the plugin
2. Open the Script Editor and run `maya/capture.py`
3. Create an instance of SculptCapture: `capture = SculptCapture()`
4. Create and select a mesh
5. Register the script job to monitor mesh changes: `capture.register_script_job()`
6. Sculpt
7. ***!!! VERY IMPORTANT !!!** Unregister the script job: `capture.unregister_script_job()` 
8. Access the recorded workflow data: `print(capture.current_workflow)`.

### Stroke parameterization
```bash
pytest tests/test_parameterization.py
```

### Neighborhood Calculation & Similarity
```bash
pytest tests/test_geodesic_calculator.py tests/test_neighborhood.py tests/test_similarity.py
```

## Suggestion Synthesis
```bash
pytest tests/test_synthesis.py
```
*TODO: add custom constraints*


