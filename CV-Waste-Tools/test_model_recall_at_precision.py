import numpy as np
from ultralytics import YOLO


def get_target_precision(default=0.95):
    while True:
        user_input = input(
            f"\nGewünschte Food-Precision in % "
            f"[Standard: {default * 100:.0f}%]: "
        ).strip()

        if not user_input:
            return default

        try:
            value = float(user_input.replace(",", "."))

            if 1 < value <= 100:
                value /= 100

            if 0 < value <= 1:
                return value

            print("Bitte einen Wert zwischen 0 und 100 eingeben.")

        except ValueError:
            print("Ungültige Eingabe. Beispiel: 95 oder 0.95")


if __name__ == "__main__":

    MODEL_PATH = (
        r"C:\Users\Public\Gerald Koinig\GreenPlast"
        r"\RECORDINGS_HIGH_RES\combined_split_lists"
        r"\training_runs\YOLO11M\weights\best.pt"
    )

    DATA_YAML = (
        r"C:\Users\Public\Gerald Koinig\GreenPlast"
        r"\RECORDINGS_HIGH_RES\combined_split_lists"
        r"\autosplit_data.yaml"
    )

    FOOD_CLASS = 0

    # User input from console
    TARGET_PRECISION = get_target_precision()

    # Load model
    model = YOLO(MODEL_PATH)
    split_type = "test"

    # Validate model
    metrics = model.val(
        data=DATA_YAML,
        split=split_type,
        conf=0.001,
        plots=True,
	workers=1
    )

    # Extract precision/recall confidence curves
    box = metrics.box

    conf = np.asarray(box.px)
    precision = np.asarray(box.p_curve[FOOD_CLASS])
    recall = np.asarray(box.r_curve[FOOD_CLASS])

    # Find thresholds meeting target precision
    valid_indices = np.where(
        precision >= TARGET_PRECISION
    )[0]

    print("\n" + "=" * 55)
    print("OPTIMIERUNG FÜR KLASSE 0: FOOD")
    print("=" * 55)

    if len(valid_indices) == 0:
        print(
            f"Ziel-Präzision von "
            f"{TARGET_PRECISION:.1%} nicht erreicht."
        )

    else:
        # Maximum recall at requested precision
        best_idx = valid_indices[
            np.argmax(recall[valid_indices])
        ]

        best_conf = conf[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]

        print(f"Ziel-Präzision:                 {TARGET_PRECISION:.1%}")
        print(f"Erreichte Precision:           {best_precision:.4f}")
        print(f"Zugehöriger RECALL:            {best_recall:.4f}")
        print(f"ERFORDERLICHER CONF-THRESHOLD:  {best_conf:.4f}")

        print("=" * 55)
        print("\nEmpfohlener Inferenzaufruf:")
        print(
            f"model.predict(source='...', "
            f"conf={best_conf:.4f})"
        )

    print("=" * 55)