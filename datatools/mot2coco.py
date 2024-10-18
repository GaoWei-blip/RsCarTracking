import os
import numpy as np
import json
import cv2
import sys
import sys
base_dir = os.getcwd()
sys.path[0] = base_dir


DATA_PATH = r'/work/RsCarData'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)


SPLITS = ['train', 'val']

for split in SPLITS:
    data_path = os.path.join(DATA_PATH, 'images', split)
    out_path = os.path.join(OUT_PATH, f'{split}.json')
    print(data_path)

    seqs = os.listdir(data_path)
    print(seqs)

    out = {'images': [], 'annotations': [], 'categories': []}
    cat = [{"supercategory": "none", "id": 1, "name": "car"}]
    out['categories'] = cat
    image_cnt = 0
    image_num = 0
    object_cnt = 0
    ann_cnt = 0
    video_cnt = 0

    for seq in seqs:
        video_cnt += 1
        seq_path = os.path.join(data_path, seq)
        print(seq_path)
        img_path = os.path.join(seq_path, 'img')
        print(img_path)
        ann_path = os.path.join(seq_path, "gt/" + seq + "-gt.csv")
        images = os.listdir(img_path)
        num_images = len([image for image in images if 'bmp' in image])
        print('num_images:', num_images)
        image_range = [0, num_images - 1]
        print(image_range)

        for k in range(num_images):
            if k < image_range[0] or k > image_range[1]:
                continue
            img = cv2.imread(os.path.join(data_path, f'{seq}/img/{str(k + 1).zfill(6)}.bmp'))
            height, width = img.shape[:2]

            image_info = {
                'file_name': f'images/{split}/{seq}/img/{str(k + 1).zfill(6)}.bmp',
                'id': int(image_cnt + image_num + 1),
                'height': int(height), 'width': int(width)
            }

            print(image_info)

            out['images'].append(image_info)
            image_num += 1

            anns = np.loadtxt(ann_path, dtype=np.int_, delimiter=',')
            print(anns)
            print(anns.shape[0])

            for i in range(0, anns.shape[0]):
                frame_id = int(anns[i][0])
                if frame_id < image_range[0] or frame_id > image_range[1]:
                    continue
                if frame_id != k:
                    continue
                track_id = int(anns[i][1])
                cat_id = int(anns[i][7])
                object_cnt += 1
                ann_cnt += 1
                category_id = int(anns[i][7])
                segmentation = [[
                    int(anns[i][2]), int(anns[i][3]),
                    int(anns[i][2]), int(anns[i][3] + anns[i][5]),
                    int(anns[i][2] + anns[i][4]), int(anns[i][3] + anns[i][5]),
                    int(anns[i][2] + anns[i][4]), int(anns[i][3])
                ]]
                ann = {
                    'segmentation': segmentation,
                    'area': float(anns[i][4] * anns[i][5]),
                    'iscrowd': 0,
                    'ignore': 0,
                    'category_id': 1,
                    'image_id': int(image_cnt + image_num),
                    'bbox': anns[i][2:6].tolist(),
                    'id': int(ann_cnt)
                }

                # {"segmentation": [[586, 866,   586, 866+8,   586+7, 866+8,  593, 866]], "area": 56, "iscrowd": 0, "ignore": 0,"image_id": 1, "bbox": [586, 866, 7, 8], "category_id": 1, "id": 4},
                out['annotations'].append(ann)
                print(ann)

        image_cnt += num_images

    # 使用 default 参数来处理 numpy 数据类型
    json.dump(out, open(out_path, 'w'), default=lambda o: int(o) if isinstance(o, np.integer) else o)
