from rle_to_poly_coordinates import rle_to_poly_coordinates
import json
import os


def cvat_to_yolo(cvat_json_file_name, path):


    #loading the json file
    json_path = os.path.join(path, cvat_json_file_name)
    print(json_path)
    f = open(json_path)
    data = json.load(f)


    #getting the image file names
    image_names = []

    for f in data['images']:
        image_names.append(f['file_name'])

    #getting the list of categories
    categories = []
    for cat in data['categories']:
        categories.append(cat['name'])

    #getting all the annotations in rle format
    annotations = data['annotations']


    #looping through all the annotations and creating the relavent text file
    text_file_names = []
    text = []
    for annotation in annotations:
        #creating the text file name and adding it to the list of text files
        text_file_name =  image_names[annotation['image_id']-1].split('.')[0]+'.txt'
        
        
        #getting the category id
        label = annotation['category_id'] - 1 

        #getting the xy list of points for the polygon

        coordinates = rle_to_poly_coordinates(annotation['segmentation']['counts'], data['images'][annotation['image_id']-1]['width'], data['images'][annotation['image_id']-1]['height'])
        string = str(label) + ' ' + coordinates

        #checking if that image already has an existing file
        if text_file_name in text_file_names:

            #if it does then append to that string
            annotation_index = text_file_names.index(text_file_name)
            text[annotation_index] = text[annotation_index] + string

        else:
            #if not create a new string for that file
            text_file_names.append(text_file_name)
            text.append(string)


    #creating the labels folder 
    os.mkdir(os.path.join(path, 'yolo_labels'))
    #after creating all the text file strings write those files to a new directory called "yolo_labels"
    for i, file_name in enumerate(text_file_names):
        #creating the file path
        file_path = os.path.join(path, 'yolo_labels', file_name)


        #getting the string for that file
        file_text = text[i]
        with open(file_path, 'w') as file:
            file.write(file_text)