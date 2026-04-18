import subprocess
import time
import sys

process = subprocess.Popen(['python3', 'interactive_train.py'], stdin=subprocess.PIPE, stdout=sys.stdout, text=True, bufsize=1)

# Supply 'y' to all prompts automatically
for _ in range(7):
    time.sleep(10)
    process.stdin.write("y\n")
    process.stdin.flush()

process.wait()
