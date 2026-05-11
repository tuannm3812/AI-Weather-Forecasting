import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_forecasting.models import train_and_save_models


if __name__ == "__main__":
    results = train_and_save_models(refresh_data=True)
    print(f"[OK] data: {results['data']}")
    for task, result in results.items():
        if task == "data":
            continue
        print(f"[OK] {task}: {result['path']}")
        for metric, value in result["metrics"].items():
            print(f"  {metric}: {value:.4f}")
