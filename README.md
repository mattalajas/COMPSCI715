# COMPSCI715
Agent for Automated Navigation in VR Environments

# Data utils

## Get data by game name
Put the game name in the `load_data_by_name` function, or leave it blank to get all game data.
```python
from utils.data_utils import DataUtils
df = DataUtils.load_data_by_name("")
```

## Import packages
Ensure you run the script from the root directory of the project to allow Python to resolve the package structure correctly. For example:
python CNNRNN/practice1.py