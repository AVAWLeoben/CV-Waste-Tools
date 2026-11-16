# v18.4.2 — Model Export: environment discovery fixes

The export environment selector used to display only the managed Studio Python
on some Windows systems even when many Conda environments were installed.

## Fixed

- Windows Conda uses `ENV_PREFIX/python.exe`, not `ENV_PREFIX/Scripts/python.exe`.
  The scanner now recognises the correct interpreter layout.
- Search the Conda installation that contains Studio, standard Anaconda/Miniconda/
  Miniforge installation paths, known environment directories, Conda's
  `~/.conda/environments.txt` registry, and custom Conda installations.
- On first visit to **07 Model Export**, perform a full Conda environment search
  in a background thread. **Find environments** repeats the scan on demand.
- Display the number of detected environments; keep the chosen interpreter
  when refreshing and persist it across restarts.
- Discard an older capability-probe result when the operator changes Python
  during the probe, then check the newly selected interpreter.
- Preserve Linux SenSoRTC's timestamped installation environments.
- Normalise `setup_yolo_export.sh` to Unix line endings, so Bash can run it.

## Use

1. Open **07 Model Export** and let environment discovery finish.
2. Open **Export Python environment**; the dropdown now contains all detected
   environment labels and full Python paths (regardless of export readiness).
3. If you installed another environment recently, click **Find environments**.
4. Choose an interpreter. Studio will check that interpreter's installed export
   dependencies without modifying them.
5. For unusual custom environments outside scanned paths, use **Browse Python…**.

The UI always runs in its managed environment. Only export probes and export
jobs run in the selected environment. No packages are installed or upgraded
by discovery. TensorRT runtime compatibility remains environment-specific.
