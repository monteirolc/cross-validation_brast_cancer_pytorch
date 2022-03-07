import os
from time import sleep

os.system("python3 -m venv .venv")
sleep(5)
os.system("source .venv/bin/activate")
sleep(5)
os.system("pip install -r requirements.txt")
