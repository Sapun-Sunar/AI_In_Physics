# Considering most Fabry_perot_Patterns are in .sip file, below is an extractor.
import sif_parser
import numpy as np
import matplotlib.pyplot as plt

def decoded_matrix(file_path):
    raw_data, info = sif_parser.np_open(file_path)
    image_matrix = np.squeeze(raw_data)
    return image_matrix

if __name__ == "__main__":
    FilePath = input("Insert your file to check:".strip("'\""))
    img = decoded_matrix(FilePath)
    plt.figure(figsize=(8,6))
    plt.imshow(img, cmap="inferno", origin='lower')
    plt.tight_layout()
    plt.savefig("fabry_perot_image",dpi=400)
    plt.show()
