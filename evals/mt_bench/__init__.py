import sys
import os

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the Python path
if current_dir not in sys.path:
    sys.path.append(current_dir)
