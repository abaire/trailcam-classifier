#!/usr/bin/env bash

set -eu -o pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <directory_to_process>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
readonly SCRIPT_DIR

cd "${SCRIPT_DIR}/.."
hatch run trailcamclassify "$1" \
      --model model/trailcam_classifier_model.pt \
      --output "~/Desktop/trail_cam/processed" \
      --copy \
      --keep-empty \
      --preserve-directories
