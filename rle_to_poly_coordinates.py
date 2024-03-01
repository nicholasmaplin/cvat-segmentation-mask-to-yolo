import mask_to_polygon
import RLE_to_mask
import json
import numpy as np


# need the RLE mask, the image width and height, 
def rle_to_poly_coordinates(rle_mask = [], image_width=0, image_height=0):
    #define the RLE object
    rle = RLE_to_mask.RLEMask(run_lengths=rle_mask, shape=(image_width, image_height))

    #decoding that object into a mask (width by heigh np array)
    mask = rle.decode()

    #next step is to turn that mask into a set of polygon points in the form [x1, y1, x2, y2, ...]
    
    #need to get the corners of the possible polygon (same as image bounds)
    corners = np.array([[0, 0], [image_width, image_height]])

    #creating the polygon object
    polygon = mask_to_polygon.Polygon(corners)

    #creating the polygon for that mask
    polygon = polygon.from_mask(mask)

    #getting the coordinates (these are in the format [[y1, x1], [y2, x2], [....]])
    coordinates = polygon.coordinates

    #generating the x and y arrays of the coordinates
    x = np.array([1-((image_width - j[1])/image_width) for j in coordinates])
    y = np.array([1-((image_height - j[0])/image_height) for j in coordinates])

    #converting them to type string
    x = x.astype(str)
    y = y.astype(str)

    #creating the final string of coordinates in the needed format for a YOLO dataset
    string = ''
    for i in range(len(x)):
        string = string  + x[i] + ' ' + y[i] + ' '
    string = string + '\n'

    #returning the string
    return string





