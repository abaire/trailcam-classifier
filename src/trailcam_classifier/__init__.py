from __future__ import annotations

import asyncio

import trailcam_classifier.import_dataset
import trailcam_classifier.main
import trailcam_classifier.training


def run() -> int:
    return asyncio.run(trailcam_classifier.main.main())


def train() -> int:
    return trailcam_classifier.training.main()


def import_dataset() -> int:
    return trailcam_classifier.import_dataset.main()
