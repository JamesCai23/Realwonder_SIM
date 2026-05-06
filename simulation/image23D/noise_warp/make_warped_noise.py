#Ryan Burgert 2024

#Setup:
#    Run this in a Jupyter Notebook on a computer with at least one GPU
#        `sudo apt install ffmpeg git`
#        `pip install rp`
#    The first time you run this it might be a bit slow (it will download necessary models)
#    The `rp` package will take care of installing the rest of the python packages for you

from huggingface_hub.inference._generated.types import video_classification
import rp
import shutil
import os

rp.r._pip_import_autoyes=True #Automatically install missing packages

rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource
import simulation.image23D.noise_warp.noise_warp as nw
import numpy as np

import cv2
import numpy as np

def resize_flow(flow, new_size=(832, 832)):
    """
    Resize a flow field of shape (2, H, W) to (2, new_H, new_W).

    Parameters:
        flow (numpy.ndarray): Flow data of shape (2, H, W).
        new_size (tuple): Desired output size (new_H, new_W).

    Returns:
        numpy.ndarray: Resized flow of shape (2, new_H, new_W).
    """
    resized_flow = np.zeros((2, new_size[0], new_size[1]), dtype=flow.dtype)

    for i in range(2):  # Resize each flow channel separately
        resized_flow[i] = cv2.resize(flow[i], new_size, interpolation=cv2.INTER_LINEAR)

    return resized_flow

class NoiseWarper:
    """
    A class for processing videos and generating warped noise for CogVideoX.
    
    Takes a video URL or filepath and converts it to warped noise at latent resolution
    with optical flow calculations.
    """
    
    def __init__(self):
        """Initialize the NoiseWarper class."""
        pass

    # def process(self, output_folder, flows, video, input_flow = True):
    #     """
    #     Takes a video URL or filepath and an output folder path
    #     It then resizes that video to height=480, width=720, 49 frames (CogVidX's dimensions)
    #     Then it calculates warped noise at latent resolution (i.e. 1/8 of the width and height) with 16 channels
    #     It saves that warped noise, optical flows, and related preview videos and images to the output folder
    #     The main file you need is <output_folder>/noises.npy which is the gaussian noises in (H,W,C) form
        
    #     Parameters:
    #         video (str): Video URL or filepath, or folder path containing optical flow files
    #         output_folder (str): Output folder path where results will be saved
    #         first_frame (str, optional): Path to first frame image
    #         sim_name (str, optional): Simulation name for saving files
    #         crop_start (int): Starting position for cropping (default: 176)
            
    #     Returns:
    #         output: The noise generation output containing numpy_noises and output_folder
    #     """

    #     FLOW = 2 ** 2
    #     LATENT = 8
    #     FRAME = 2**-1
       
    #     if input_flow:
    #         output = nw.get_noise_from_video(
    #             flows,
    #             remove_background=False, #Set this to True to matte the foreground - and force the background to have no flow
    #             visualize=True,          #Generates nice visualization videos and previews in Jupyter notebook
    #             save_files=True,         #Set this to False if you just want the noises without saving to a numpy file
    #             input_flow=True,
    #             noise_channels=32,
    #             output_folder=output_folder,
    #             resize_frames=FRAME,
    #             resize_flow=FLOW,
    #             input_flow=input_flow,
    #             downscale_factor= round(FRAME * FLOW) * LATENT,
    #         )
    #     else:
    #         video = rp.as_numpy_array(video)

    #         output = nw.get_noise_from_video(
    #             video,
    #             remove_background=False, #Set this to True to matte the foreground - and force the background to have no flow
    #             visualize=True,          #Generates nice visualization videos and previews in Jupyter notebook
    #             save_files=True,         #Set this to False if you just want the noises without saving to a numpy file
    #             input_flow=False,
    #             noise_channels=32,
    #             output_folder=output_folder,
    #             resize_frames=FRAME,
    #             resize_flow=FLOW,
    #             input_flow=input_flow,
    #             downscale_factor= round(FRAME * FLOW) * LATENT,
    #         )

        

    #     print("Noise shape:"  ,output.numpy_noises.shape)
    #     print("Output folder:",output.output_folder)
        
    #     return output.numpy_noises
    

    def process(self, video, output_folder:str, input_flow = True, crop_start = 120, debug = False, device = None):

        FLOW = 2 ** 3
        LATENT = 8

        if input_flow:
            frame_flows = video
            FRAME = 1
            FLOW = 2 ** 2

            if debug:
                print("Number of frames for optical flow:", len(frame_flows))
                print("Shape of each frame:", frame_flows[0].shape)

            resized_flows = []
            for i in range(len(frame_flows)):
                resized_flow = resize_flow(frame_flows[i])
                resized_flows.append(resized_flow)
            frame_flows = resized_flows

            for i in range(len(frame_flows)):
                frame_flows[i] = frame_flows[i] * (832 / 512)
            for i in range(len(frame_flows)):
                frame_flows[i] = frame_flows[i][:, crop_start:crop_start + 480, :]
            frame_flows = rp.as_numpy_array(frame_flows)
            if debug:
                print("Shape of optical flow:", frame_flows.shape)
            video = frame_flows
            input_flow = True

        else:
            FRAME = 2**-1
            video = rp.as_numpy_array(video)

        output = nw.get_noise_from_video(
            video,
            remove_background=False,
            visualize=debug,
            save_files=True,
            input_flow=input_flow,
            noise_channels=32,
            output_folder=output_folder,
            resize_frames=FRAME,
            resize_flow=FLOW,
            downscale_factor= round(FRAME * FLOW) * LATENT,
            device=device,
        )

        if debug:
            print("Noise shape:"  ,output.numpy_noises.shape)
            print("Output folder:",output.output_folder)