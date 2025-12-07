#!/usr/bin/env bash
set -eu


KNOWN_NAMES=(
  "/Volumes/NO NAME"
  "/Volumes/MOUTRIECAM"
)

TARGET_ROOT="/Volumes/Transfer"

if [[ ! -d "${TARGET_ROOT}" ]]; then
    echo "Error: Target root directory '${TARGET_ROOT}' does not exist." >&2
    exit 1
fi

TODAYS_ISO_DATE=$(date +%Y%m%d)
TARGET_PARENT="${TARGET_ROOT}/unclassified_${TODAYS_ISO_DATE}"
if [[ ! -d "${TARGET_PARENT}" ]]; then
    mkdir -p "${TARGET_PARENT}"
fi


function transfer() {
  local path="${1}"

  echo "Processing source: ${path}"
  
  # Check if DCIM exists before proceeding
  local source_dcim="${path}/DCIM"
  if [[ ! -d "${source_dcim}" ]]; then
      echo "Error: No DCIM directory found at '${path}'."
      return
  fi


  local counter=1
  local target_dir=""

  while true; do
      local candidate="${TARGET_PARENT}/${counter}"
      if [[ ! -d "${candidate}" ]]; then
          target_dir="${candidate}"
          mkdir -p "${target_dir}"
          echo "Created batch directory: ${target_dir}"
          break
      fi
      ((counter++))
  done


  find "${source_dcim}" -type f -iname "*.jpg" -print0 | while IFS= read -r -d '' file; do
      mv "${file}" "${target_dir}/"
      
      # Check if the move command succeeded
      if [[ $? -ne 0 ]]; then
          echo "Error: Failed to move file '${file}'. Aborting operation." >&2
          exit 1
      fi
  done

  rm -rf "${source_dcim}"

  echo "Transfer complete, unmounting ${path}..."
  if diskutil unmount "${path}"; then
      echo "Successfully unmounted ${path}. It is safe to remove."
  else
      echo "Error: Failed to unmount ${path}." >&2
  fi
}


for name in "${KNOWN_NAMES[@]}"; do
  if [[ -d "${name}" ]]; then
    transfer "${name}"
  fi
done
