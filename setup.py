# setup.py
import os
import sys
import subprocess
from setuptools import setup, find_packages

# Check if venv exists, create if it doesn't
venv_dir = "venv"
if not os.path.exists(venv_dir):
    subprocess.call([sys.executable, "-m", "venv", venv_dir])

# Get the path to the venv's pip
if os.name == "nt":  # Windows
    pip_path = os.path.join(venv_dir, "Scripts", "pip")
else:  # Unix/Linux/MacOS
    pip_path = os.path.join(venv_dir, "bin", "pip")

# Install requirements
subprocess.call([pip_path, "install", "-r", "requirements.txt"])

setup(
    name="speech-to-text-visualizer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Flask",
        "matplotlib",
        "numpy",
        "gunicorn",
    ],
)

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

print(
    "Setup complete! To run the app, activate the virtual environment and run 'python app.py'"
)