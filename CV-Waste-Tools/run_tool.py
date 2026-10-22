# -*- coding: utf-8 -*-
"""Start the selected GUI; the launcher captures stdout/stderr in log files."""
import os
from pathlib import Path
import sys
import traceback

os.chdir(Path(__file__).resolve().parent)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "audit", "recall"):
        raise SystemExit("Usage: python run_tool.py [train|audit|recall]")
    if sys.argv[1] == "train":
        from train_and_split_tool_GUI import autosplit_and_train_model
        autosplit_and_train_model()
    elif sys.argv[1] == "audit":
        from audit_dataset_review_gui import main as audit_main
        audit_main()
    else:
        from test_model_recall_at_precision import main as recall_main
        recall_main()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("\nThe application failed. Inspect the traceback above.")
        if sys.stdin.isatty():
            try:
                input("Press Enter to close this console ...")
            except EOFError:
                pass
        raise SystemExit(1)
