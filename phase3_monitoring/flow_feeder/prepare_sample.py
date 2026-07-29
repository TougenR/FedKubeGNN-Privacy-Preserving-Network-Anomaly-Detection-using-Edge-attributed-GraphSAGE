import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

SOURCE_FILE = os.path.join(_REPO_ROOT, "data", "CTU-IoT-Malware-Capture-34-1", "conn.log.labeled")
DEST_FILE = os.path.join(_THIS_DIR, "sample_data", "sample_conn.log.labeled")

def create_sample(num_lines=400):
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Source file not found: {SOURCE_FILE}")
        sys.exit(1)

    os.makedirs(os.path.dirname(DEST_FILE), exist_ok=True)

    lines_written = 0
    with open(SOURCE_FILE, 'r', encoding='utf-8') as src:
        with open(DEST_FILE, 'w', encoding='utf-8') as dest:
            for i, line in enumerate(src):
                if i >= num_lines:
                    break
                dest.write(line)
                lines_written += 1

    print(
        f"Successfully created sample data at {DEST_FILE} "
        f"({lines_written} lines)."
    )

if __name__ == "__main__":
    create_sample()
