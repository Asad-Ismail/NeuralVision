import json
import os

def divide_coco_json(input_json_file, output_directory):
    # Read the input JSON file
    with open(input_json_file, 'r') as infile:
        coco_data = json.load(infile)

    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through each image in the COCO dataset
    for image in coco_data['images']:
        # Create a new JSON structure for the current image
        image_json = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'categories': coco_data['categories'],
            'images': [image],
            'annotations': [],
        }

        # Get the annotations for the current image
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image['id']:
                image_json['annotations'].append(annotation)

        # Save the new JSON file for the current image
        output_file = os.path.join(output_directory, f"{image['file_name'].split('.')[0]}.json")
        with open(output_file, 'w') as outfile:
            json.dump(image_json, outfile, indent=2)

    print(f"JSON files created in the '{output_directory}' directory.")


# Example usage
input_json_file = '~/Downloads/Balloons.v15i.coco-segmentation/train/_annotations.coco.json'
output_directory = '~/Downloads/Balloons.v15i.coco-segmentation/train/annoations'
divide_coco_json(input_json_file, output_directory)
