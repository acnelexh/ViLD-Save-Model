import json
from pathlib import Path
import pdb


def combine(json_path, out_path):
    result = []
    #pdb.set_trace()
    files_list = [x for x in json_path.glob("*.json")]
    for file in files_list:
        with open(file, 'r') as f:
            image_detection = json.load(f)
            for detection in image_detection:
                score = detection['score']
                if score < 0.5:
                    continue
                #pdb.set_trace()
                x1,y1,x2,y2 = detection['bbox']
                x = (x1+x2)/2
                y = (y1+y2)/2
                width = x2-x1
                height = y2-y1
                modified_dict = {
                    'image_id': detection['image_id'],
                    'category_id': detection['score'],
                    'bbox': [x, y, width, height],
                    'score': score
                }
                result.append(modified_dict)
    pdb.set_trace()
    with open(out_path, 'w') as outfile:
        json_object = json.dumps(result, indent = 2)
        outfile.write(json_object)
    #pdb.set_trace()

def check_result(out_path):
    with open(out_path) as f:
        result = json.load(f)
    print(len(result))


combine(Path("result_new"), Path("./result_new.json"))
#check_result(Path("./result_new.json"))