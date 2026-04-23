import subprocess
import time
import sys

process = subprocess.Popen(['python3', 'interactive_train.py'], stdin=subprocess.PIPE, stdout=sys.stdout, text=True, bufsize=1)

# Supply 'y' to all 8 prompts automatically
for i in range(8):
    print(f"Sending 'y' for step {i+1}...")
    time.sleep(15) 
    process.stdin.write("y\n")
    process.stdin.flush()

process.wait()
