"""Run TERMO for the week, from the repository root.

    python run_termo.py --download [--mes AAAA-MM]
    python run_termo.py --snapshot data/snapshots/<fecha> [--mes AAAA-MM]

The deliverable lands under output/<reading date>/ (reporte.html plus the files it was
built from). See `python run_termo.py --help`.
"""

from termo.run_week import main

if __name__ == "__main__":
    raise SystemExit(main())
