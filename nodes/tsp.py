# via https://github.com/dmarx/video-killed-the-radio-star/blob/main/vktrs/tsp.py

import time

import numpy as np
from scipy.spatial.distance import pdist, squareform
from toolz.itertoolz import partition_all
from python_tsp.exact import solve_tsp_dynamic_programming


# def tsp_permute_frames(frames, verbose=False):
#     """
#     Permutes images using traveling salesman solver to find frame-to-frame 
#     ordering that minimizes difference between subsequent frames, i.e.
#     the ordering of the images that gives the smoothest animation.
#     """
#     frames_m = np.array([np.array(f).ravel() for f in frames])
#     logger.debug(len(frames_m))
#     logger.debug(frames_m[0].shape)
#     dmat = pdist(frames_m, metric='cosine')
#     dmat = squareform(dmat)

#     start = time.time()
#     permutation, _ = solve_tsp_dynamic_programming(dmat)
#     if verbose:
#         print(f"elapsed: {time.time() - start}")

#     frames_permuted = [frames[i] for i in permutation]
#     return frames_permuted

# from vktrs notebook
def tsp_sort(frames):
    frames_m = np.array([np.array(f).ravel() for f in frames])
    dmat = pdist(frames_m, metric='cosine')
    dmat = squareform(dmat)
    permutation, _ = solve_tsp_dynamic_programming(dmat)
    return permutation

# def batched_tsp_permute_frames(frames, batch_size):
#     """
#     TSP solver is O(n^2). Instead of limiting how many variations a user can
#     request for a particular image, we set an upperbound on how many images we
#     send to the solver at any given time. TODO: Faster solver.
#     """
#     ordered = []
#     for batch in partition_all(batch_size, frames):
#         ordered.extend( tsp_permute_frames(batch) )
#     return ordered

def batched_tsp_sort(frames, batch_size):
    """
    TSP solver is O(n^2). Instead of limiting how many variations a user can
    request for a particular image, we set an upperbound on how many images we
    send to the solver at any given time. TODO: Faster solver.
    """
    order = []
    for batch in partition_all(batch_size, frames):
        prev=len(order)
        for j in tsp_sort(batch):
            order.append(prev+j)
    return order

####

#from loguru import logger
import torch 

CATEGORY="tsp"

class TSPPermuteFrames:
    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES=("IMAGE",)
    
    @classmethod
    def INPUT_TYPES(cls):
        outv = {
            "required": {
                "images": ("IMAGE",{"forceInput": True,}),
                "batch_size": ("INT",{"default":12}),
            },
        }
        return outv

    def main(self, images, batch_size=12):
        #images = batched_tsp_permute_frames(images, batch_size)
        #idx = tsp_sort(images)
        idx = batched_tsp_sort(images, batch_size)
        idx = torch.tensor(idx, device=images.device)
        #outv = torch.cat(images, dim=0)
        outv = images[idx]
        #logger.debug(outv.shape)
        return (outv,)


NODE_CLASS_MAPPINGS = {
    "TSPPermuteFrames": TSPPermuteFrames
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TSPPermuteFrames": "TSPPermuteFrames"
}