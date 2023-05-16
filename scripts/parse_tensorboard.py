import json
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


output_metrics = {
    "params": {
        "dropout": 0.2
    }
}

target_logs = [
    ("refined", "checkpoints/refined-May16_10-04-53/logs/events.out.tfevents.1684231493.gpu-t4-big-disk-1.25561.0")
]

for tag, logs_path in target_logs:
    event_acc = EventAccumulator(logs_path)
    event_acc.Reload()

    val_miou = np.array([x.value for x in event_acc.Scalars("val/miou")])

    output_metrics[tag] = {
        "val_miou": val_miou.tolist(),
        "best_epoch": int(val_miou.argmax()),
        "train_miou": [x.value for x in event_acc.Scalars("train/miou")]
    }

with open("refined_metrics.json", "w") as out_file:
    out_file.write(json.dumps(output_metrics))
