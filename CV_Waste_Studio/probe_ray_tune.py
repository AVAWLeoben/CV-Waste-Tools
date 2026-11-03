"""Windows-safe subprocess probe for optional Ray/Optuna tuning dependencies."""
import json
import multiprocessing
from ray_tune_support import dependency_report

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print(json.dumps(dependency_report()))
