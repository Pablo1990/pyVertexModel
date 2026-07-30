import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path(
    os.environ.get(
        "PYVERTEXMODEL_OUTPUT_ROOT",
        PROJECT_DIR / "Result" / "aspect_ratio_cell_height_ablation_sweep",
    )
)
LOG_DIR = OUTPUT_ROOT / "logs"
PARAMETER_INDICES = range(105)
MAX_JOBS = 3


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    running = []
    failures = []

    for idx in PARAMETER_INDICES:
        while len(running) >= MAX_JOBS:
            process, running_idx = running.pop(0)
            return_code = process.wait()
            process._pyvertexmodel_log.close()
            if return_code != 0:
                print(f"[{time.ctime()}] Sweep job {running_idx} failed with code {return_code}")
                failures.append(return_code)
            else:
                print(f"[{time.ctime()}] Finished sweep job {running_idx}")

        log_file = LOG_DIR / f"sweep_job_{idx}.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PYVERTEXMODEL_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYVISTA_OFF_SCREEN"] = "true"

        print(f"[{time.ctime()}] Starting sweep job {idx}")
        log = log_file.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pyVertexModel.run_cyst_aspect_cell_height",
                str(idx),
            ],
            cwd=PROJECT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        process._pyvertexmodel_log = log
        running.append((process, idx))

    for process, idx in running:
        return_code = process.wait()
        process._pyvertexmodel_log.close()
        if return_code == 0:
            print(f"[{time.ctime()}] Finished sweep job {idx}")
        else:
            print(f"[{time.ctime()}] Sweep job {idx} failed with code {return_code}")
            failures.append(return_code)

    if failures:
        sys.exit(failures[0])

    print(f"[{time.ctime()}] All aspect-ratio/cell-height/ablation simulations completed.")


if __name__ == "__main__":
    main()
