import subprocess
import sys
import termios
import tty
import os

SCRIPT = "./prepare_and_run_gui.sh"  # adjust path if needed

def get_key():
    """Read a single keypress from stdin (works in WSL / Linux terminal)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    typing_started = False
    print("Press any key to start typing. Press Enter to run the script.")
    while True:
        key = get_key()
        if key == '\r' or key == '\n':
            print("\nEnter pressed! Running script...\n")
            subprocess.run(["bash", SCRIPT])
            typing_started = False
            print("\nPress any key to start typing again. Press Enter to run the script.")
        else:
            if not typing_started:
                print("Typing started...")
                typing_started = True

if __name__ == "__main__":
    main()