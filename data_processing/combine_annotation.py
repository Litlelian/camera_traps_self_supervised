import json
import os

final_annotation = {'images':[], 'annotations':[]}
files_to_process = os.listdir('./cam_data/eccv_18_annotation_files/')
for files in files_to_process:
    filepath = './cam_data/eccv_18_annotation_files/' + files
    with open(filepath) as f:
        current_json = json.load(f)
    final_annotation['images'] += current_json['images']
    final_annotation['annotations'] += current_json['annotations']

with open('./cam_data/eccv_18_annotation_files/CaltechCameraTrapsECCV18.json', 'w') as fp:
    json.dump(final_annotation, fp)