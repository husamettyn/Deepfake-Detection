#!/bin/bash

# The name of your current dataset folder
SOURCE_DIR="FaceForensics++_C23"

# The name of the new folder that the code expects
TARGET_DIR="Faceforensics"

echo "Creating target directory structure..."
mkdir -p "${TARGET_DIR}/manipulated_sequences"
mkdir -p "${TARGET_DIR}/original_sequences/youtube/c23"

# --- Process Manipulated Videos ---
# List of manipulation folders to process
MANIPULATIONS=("DeepFakeDetection" "Deepfakes" "Face2Face" "FaceSwap" "NeuralTextures" "FaceShifter")

echo "Processing manipulated sequences..."
for manip in "${MANIPULATIONS[@]}"; do
    if [ -d "${SOURCE_DIR}/${manip}" ]; then
        echo "  -> Moving ${manip}..."
        # Create the target subdirectory structure
        mkdir -p "${TARGET_DIR}/manipulated_sequences/${manip}/c23"
        # Move the source folder and rename it to 'videos'
        mv "${SOURCE_DIR}/${manip}" "${TARGET_DIR}/manipulated_sequences/${manip}/c23/videos"
    else
        echo "  -> Skipping ${manip} (not found)."
    fi
done

# --- Process Original Videos ---
echo "Processing original sequences..."
if [ -d "${SOURCE_DIR}/original" ]; then
    # Move the 'original' folder and rename it to 'videos'
    mv "${SOURCE_DIR}/original" "${TARGET_DIR}/original_sequences/youtube/c23/videos"
else
    echo "  -> 'original' folder not found."
fi

echo "Reorganization complete! Your data is now in the '${TARGET_DIR}' folder."
echo "You can now delete the empty '${SOURCE_DIR}' folder if you wish."