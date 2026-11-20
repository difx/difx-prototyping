# results/

Benchmark summaries, small enough to keep in the repository: one CSV row per shape.  The logs and
the full per-run CSVs (`fusebench --csv`: every timing and check) stay in `results-local/`, which is
not committed.

- `<host>-matrix-<stamp>.csv`: `run-matrix.sh`, the whole chain (station processing + XMAC) over
  bits x subbands x channels x stations.  Columns: see `matrix_summary.py`.  Times are ms per ms of
  data; compare absolute times only within one GPU and one power/clock setting (`power_w`,
  `sm_clock_lock_mhz`; ar313's RTX 2070 SUPER is capped at 150 W and, from 2026-11-20, 1500 MHz).
