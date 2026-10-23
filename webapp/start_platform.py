"""Start the local platform with the isolated, pinned Python environment."""
from pathlib import Path
import os
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "scratch/c3-py313/Scripts/python.exe"


def listening(port):
    with socket.socket() as connection:
        connection.settimeout(.25)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def start(name, command, directory, environment):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (ROOT / f"scratch/{name}.log").open("ab") as output:
        process = subprocess.Popen(command, cwd=directory, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=output, creationflags=flags)
    print(f"Started {name} (PID {process.pid}); log: scratch/{name}.log")


def main():
    if not PYTHON.is_file():
        raise SystemExit("Pinned environment missing. See component3/docs/SYSTEM_REPAIR_REPORT.md.")
    (ROOT / "scratch").mkdir(exist_ok=True)
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.setdefault(variable, "2")
    if not listening(8000):
        start("platform-backend", [str(PYTHON), "-m", "uvicorn", "app.main:app", "--app-dir",
              "webapp/backend", "--host", "127.0.0.1", "--port", "8000"], ROOT, environment)
    else:
        print("Port 8000 is already serving; left the existing process unchanged.")
    if not listening(5173):
        node = shutil.which("node")
        if not node:
            raise SystemExit("Node.js is required for the frontend.")
        start("platform-frontend", [node, "node_modules/vite/bin/vite.js", "--host", "127.0.0.1",
              "--port", "5173", "--strictPort"], ROOT / "webapp/frontend", environment)
    else:
        print("Port 5173 is already serving; left the existing process unchanged.")
    print("Dashboard: http://127.0.0.1:5173/c3 ; API: http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    main()
