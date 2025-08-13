from __future__ import annotations

import trailcam_classifier.main
import trailcam_classifier.training


def run() -> int:
    return trailcam_classifier.main.main()


def train() -> int:
    return trailcam_classifier.training.main()
