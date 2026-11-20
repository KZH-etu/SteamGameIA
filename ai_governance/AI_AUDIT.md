# AI Audit

<!-- Modèle d'entrée :
### Flaw N – <titre> (Module X – <prénom>)
- **AI output:** what the AI generated (code snippet)
- **Problem:** what is wrong + root cause (e.g. scaler fitted on full data → leakage)
- **How we detected it:** test, assert, metric that looked too good...
- **Fix:** corrected code
- **Impact:** metric before / after the fix
-->

### Flaw 1 – Full review file kept in memory → kernel crash (Module 1 – Yohan ALLOUCHE)
- **AI output:** the file was read "in chunks", but every chunk was kept and concatenated, then duplicates were searched on the whole table:
  ```python
  chunks = []
  for chunk in pd.read_csv(path, usecols=COLS + ["review_id"], dtype=DTYPES, chunksize=2_000_000):
      chunk["date"] = pd.to_datetime(chunk["date"], format="%Y-%m-%d", errors="coerce")
      chunks.append(chunk)
  reviews = pd.concat(chunks, ignore_index=True)   # 41M rows
  dup_rows = reviews.duplicated(subset=COLS).sum()
  ```
- **Problem:** reading in chunks only saves memory if each chunk is reduced before being kept. Here all 41M rows end up in memory: `pd.concat` holds the chunks and the result at the same time (≈ 2 × 0.9 GB), and `duplicated()` builds large intermediate arrays (estimated ≈ 1.5 GB). With about 4 GB of free RAM, the kernel died. Root cause: the AI applied "read the file in chunks" mechanically, without filtering inside the loop.
- **How we detected it:** the notebook stopped with `DeadKernelError: Kernel died` after 82 seconds.
- **Fix:** for the inspection, a development sample (`nrows=5_000_000`). For `reviews_clean`, each chunk is filtered on the sampled games before being kept:
  ```python
  for chunk in pd.read_csv(..., chunksize=2_000_000):
      chunks.append(chunk[chunk["app_id"].isin(release.index)])
  ```
- **Impact:** before: crash, no output. After: the notebook runs; only the reviews of the 10,000 sampled games stay in memory.
