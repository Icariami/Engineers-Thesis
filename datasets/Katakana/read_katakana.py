import struct
from PIL import Image
import numpy as np

# Create a mapping from the old labels (specific ones) to the new range (0 to 47)
def create_label_mapping():
    # Define the unique labels you have in the dataset (labels from your distribution)
    old_labels = [166, 168, 170, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
                  187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                  200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
                  213, 214, 215, 216, 217, 218, 219, 220, 221]

    # Create the new labels (0 to 47) mapping
    new_labels = list(range(48))
    # Create the mapping dictionary (old label -> new label)
    return dict(zip(old_labels, new_labels))

"""
>H: 2-byte unsigned short (serial number)
2s: 2-byte string (writer ID)
H: 2-byte unsigned short (character code).
6H: 6 short integers (unused metadata).
B: 1 byte (quality byte).
I: 1 integer (some identifier).
4H: 4 short integers (more metadata).
4B: 4 bytes (bitmap size and other info).
x: unused padding byte.
2016s: 2016-byte string that represents the bitmap data (this is the pixel data).
4x: extra unused bytes at the end.
"""
def read_record_ETL1G(f, label_mapping):
    s = f.read(2052)

    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)

    # Convert bitmap data to image
    # mode 'F' - floating point
    #  r[18] - starts from bit 18
    # bit - stored as bitmap
    #  4  - 4 bits per pixel
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')  # Convert the image to a palette-based image (8-bit)

    original_label = r[3]

    # Apply label normalization using the label_mapping
    normalized_label = label_mapping.get(original_label)  # Map the label to the new range

    return np.array(iL), normalized_label

# Function to read the entire dataset and save it
def read_kana():
    images = []
    labels = []

    # Create the label mapping (old labels to new labels)
    label_mapping = create_label_mapping()

    for i in range(7, 14):
        filename = '../datasets/Katakana/ETL1C_{:02d}'.format(i)
        with open(filename, 'rb') as f:
            limit = 8 if i != 13 else 3
            for dataset in range(limit):
                for j in range(1411):
                    try:
                        image, label = read_record_ETL1G(f, label_mapping)
                        images.append(image)
                        labels.append(label)
                        print(label)
                    except struct.error:  # Ignore blank images
                        pass

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint16)

    np.savez_compressed("katakana_dataset.npz", images=images, labels=labels)
    print("Dataset saved as katakana_dataset.npz")

read_kana()
