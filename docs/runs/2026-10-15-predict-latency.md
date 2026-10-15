# predict() micro-benchmark (2026-10-15)

Verified (run 2026-10-15, AMD Ryzen 7 5800X (8 cores, 16 threads), 32 GB RAM, Windows 11 Pro).

Single process, no HTTP, 300 timed calls per cell after one warm-up call; scikit-learn 1.9.1; Windows-10-10.0.26200-SP0. The machine is shared with other jobs (not controlled during this run), so absolute values are indicative; the ratios between rows are what matter. 'default' is scikit-learn's OpenMP thread count (all logical CPUs).

Command: `python loadtest/bench_predict.py --hardware "<description>"` (trains the two models on the same 80/20 split with seed 42, `load_california` data cached under `.cache/`).

| Boosting iterations | OpenMP threads | Rows per call | Median ms | p95 ms |
|---|---|---|---|---|
| 300 | default | 1 | 16.386 | 41.088 |
| 300 | default | 32 | 16.054 | 34.946 |
| 300 | 1 | 1 | 4.266 | 6.918 |
| 300 | 1 | 32 | 4.349 | 5.458 |
| 100 | default | 1 | 5.434 | 16.156 |
| 100 | default | 32 | 4.323 | 13.141 |
| 100 | 1 | 1 | 1.413 | 2.049 |
| 100 | 1 | 32 | 1.460 | 1.749 |

Reading: for one to 32 rows the cost is dominated by per-iteration overhead, not by the number of rows, and scikit-learn's default threading makes tiny calls several times slower than a single thread. The service therefore pins prediction to one thread (`MLS_PREDICT_THREADS=1`) and scales with worker processes.
