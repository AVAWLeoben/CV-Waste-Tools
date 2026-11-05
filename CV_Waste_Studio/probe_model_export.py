"""Probe model-export dependencies within CV Waste Studio's managed Python."""
import json
from model_export import probe_export_environment

if __name__ == "__main__":
    print("STUDIO_EXPORT_PROBE=" + json.dumps(probe_export_environment()), flush=True)
