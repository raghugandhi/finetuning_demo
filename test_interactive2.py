import subprocess
import time
import sys

process = subprocess.Popen(['python3', 'interactive_test.py'], stdin=subprocess.PIPE, stdout=sys.stdout, text=True, bufsize=1)

time.sleep(10)
process.stdin.write("You missed the deadline again. This is unacceptable.\n")
process.stdin.flush()

time.sleep(15)
process.stdin.write("quit\n")
process.stdin.flush()

process.wait()

