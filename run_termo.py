"""Run TERMO for the week, from the repository root.

    python run_termo.py --download [--mes AAAA-MM]
    python run_termo.py --snapshot data/snapshots/<fecha> [--mes AAAA-MM]
    python run_termo.py --solo-reporte output/<fecha> --snapshot data/snapshots/<fecha>

The deliverable lands under output/<reading date>/ (reporte.html plus the files it was
built from). The third form only rebuilds output/<fecha>/reporte.html, without running the
model; an optional output/<fecha>/comentario.md is added as the analyst's comment. See
`python run_termo.py --help`.
"""

from termo.run_week import main

if __name__ == "__main__":
    raise SystemExit(main())
