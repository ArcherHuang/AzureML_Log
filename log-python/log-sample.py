import cv2
import numpy as np
from azureml.core import Run
from azureml.core import Workspace

# 取得 Workspace
run = Run.get_context()
ws = run.experiment.workspace

# Use log
accuracy = 168
run.log("[Training Step] accuracy", accuracy)

# Use log_list
run.log_list("epoch", [50, 100, 66])

# Use log_row
run.log_row("Y over X", x=1, y=0.5)
run.log_row("Y over X", x=2, y=6.8)

# Use log_table
logtable = {
                'epochs': [1, 2, 3, 4, 5],
                'test_accuracy': [0.2, 0.3, 0.45, 0.6, 0.61],
                'train_accuracy': [0.15, 0.28, 0.41, 0.5, 0.59]
        }
run.log_table('Training curve', logtable)

# Use log_image
rgb = np.random.randint(255, size=(900, 800, 3), dtype=np.uint8)
cv2.imwrite("outputs/Image.png", rgb)
run.log_image('overfitting_monitor_img', "outputs/Image.png")

