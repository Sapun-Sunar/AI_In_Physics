import torch
import numpy as np
from sip_decoder import decoded_matrix
from brain import Spectroscopic_Brain

def find_absorption_line_width(input_matrix, model_weights_path="spectroscopic_brain_hybrid.pth"):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA Cuda.")
    else:
        device = torch.device("cpu")
        print("No GPU Found, defaulting to cpu.")


    # Instantiate the class.
    brain = Spectroscopic_Brain().to(device)


    # Unpack the checkpoint dictionary.
    try:
        # 1. Load the entire checkpoint dictionary.
        checkpoint = torch.load(model_weights_path, map_location=device)
        
        # 2. Extract ONLY the neural network weights ("model_state_dict")
        if 'model_state_dict' in checkpoint:
            brain.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback in case it actually IS just a raw state_dict
            brain.load_state_dict(checkpoint)
            
        brain.eval() # Lock the network into inference mode
        

    except FileNotFoundError:
        print("Critical Error: No trained model checkpoint found. Train one first.")
        return
    except Exception as e:
        print(f"PyTorch Loading Error: {e}")
        return
    


    raw_lab_data = input_matrix
    # Matrix crop (Cropping the matrix of the image such that only the first ring exists could result in higher accuracy).
    print("\nDefine your Fabry-Perot interference crop boundaries:")
    a = int(input("Row start: "))
    b = int(input("Row end: "))
    c = int(input("Col start: "))
    d = int(input("Col end: "))
    
    cropped_data = raw_lab_data[a:b, c:d]

    data_min = np.min(cropped_data)
    data_max = np.max(cropped_data)
    

    # Mathematical failsafe: Prevent division by zero if the crop is perfectly black/flat.
    if data_max == data_min:
        print("Critical Error: The cropped matrix has zero variance. Recalibrate your crop coordinates.")
        return
    normalized_data = (cropped_data - data_min) / (data_max - data_min)


    # Convert the 2D matrix into a 4D tensor (Batch Size 1, Channels 1, Height, Width)
    tensor_data = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = brain(tensor_data)
        fwhm_result = prediction.item()
    
    return fwhm_result



if __name__ == "__main__":
    lab_data_path = input("Enter the exact path to your .sif lab data: ")
    
    try:
        matrix = decoded_matrix(lab_data_path)
    except Exception as e:
        print(f"Matrix Extraction Failed. Ensure the sip_decoder is functioning. Error: {e}")
        exit() # Stop execution if we have no matrix

    # Run the neural network outside the decoder's exception block
    fwhm = find_absorption_line_width(matrix)
    if fwhm is not None:
        print(f"\n[TARGET ACQUIRED] The width of the absorption line is: {fwhm}")