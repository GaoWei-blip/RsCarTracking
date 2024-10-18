import cvtools.data_augs as augs
import sys
import os
sys.path.append('track2_baseline')

import sys
base_dir = os.getcwd()
sys.path[0] = base_dir


SPLITS = ['train', 'val']

for split in SPLITS:
    img_prefix = '/work/RsCarData'
    ann_file = f'/work/RsCarData/annotations/{split}.json'
    if split == 'train':
        size = 512
    elif split == 'val':
        size = 1024

    print(img_prefix)
    print(ann_file)

    # 用于裁剪的数据集中间表示层，继承自cvtools.data_augs.crop.crop_abc.CropDataset
    dataset = augs.CocoDatasetForCrop(img_prefix, ann_file)
    print(type(dataset))

    # 定义滑动窗口裁剪方法，该方法将在图像上以滑动窗口的方式依次裁剪出多个小块，重叠比例为overlap
    crop_method = augs.CropImageInOrder(crop_w=size, crop_h=size, overlap=0.2)

    # 将数据集和裁剪方法传入通用裁剪类CropLargeImages
    crop = augs.CropLargeImages(dataset, crop_method)
    crop.crop_for_train()  # 执行裁剪
    crop.save(to_file=ann_file)  # 将裁剪后的图像信息保存回指定的 JSON 文件中
