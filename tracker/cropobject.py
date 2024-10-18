import shutil
import cv2
import os
import sys
import random
import pandas as pd

base_dir = os.getcwd()
sys.path[0] = base_dir


def process_video(video_folder, save_folder):
    # 列出文件夹中的所有文件
    files = os.listdir(video_folder)

    # 初始化字典来存储配对文件
    file_pairs = {}

    # 遍历所有文件并配对
    for file in files:
        if file.endswith('.avi'):
            base_name = file[:-4]  # 去掉扩展名后的文件名
            csv_name = f"{base_name}-gt.csv"
            if csv_name in files:
                file_pairs[base_name] = {
                    'avi': os.path.join(video_folder, file),
                    'csv': os.path.join(video_folder, csv_name)
                }

    start_id = 0
    for base_name, pair in file_pairs.items():
        print(f'start_id:{start_id}')

        # 视频和CSV文件路径
        video_file_path = pair['avi']
        csv_file_path = pair['csv']

        # 读取CSV文件
        df = pd.read_csv(csv_file_path, header=None)
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', '0', '1', '2', '3']

        # 打开视频文件
        cap = cv2.VideoCapture(video_file_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            # 获取当前帧号
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # 从CSV中获取对应帧号的对象数据
            objects_in_frame = df[df['frame'] == frame_number]

            for index, row in objects_in_frame.iterrows():
                # 左上角为坐标原点
                obj_id = row['id']+start_id
                x1 = row['x']
                y1 = row['y']
                x2 = x1 + row['w']
                y2 = y1 + row['h']

                if (0 <= x1 < frame_width and 0 <= y1 < frame_height and 0 <= x2 < frame_width and
                        0 <= y2 < frame_height):
                    # 裁剪目标区域
                    cropped_img = img[y1:y2, x1:x2, :]
                    # 为每个目标创建文件夹
                    obj_folder = os.path.join(save_folder, str(obj_id))
                    if not os.path.exists(obj_folder):
                        os.makedirs(obj_folder)

                    # 保存裁剪后的目标图像
                    cropped_img_name = f"{obj_id}_{frame_number}.jpg"
                    cropped_img_path = os.path.join(obj_folder, cropped_img_name)
                    cv2.imwrite(cropped_img_path, cropped_img)
        cap.release()

        start_id = start_id + df['id'].max() + 1

        print(f"{base_name} processed")

    print("All videos processed")


def split_dataset(source_dir, train_dir, val_dir):
    # 创建目标目录（如果不存在）
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取源目录下的所有对象文件夹
    object_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    # 设置训练集与验证集的划分比例，比如 80% 训练集，20% 验证集
    train_ratio = 0.8

    # 遍历每个对象文件夹
    for folder in object_folders:
        # 获取当前对象文件夹路径
        folder_path = os.path.join(source_dir, folder)

        # 列出对象文件夹中的所有文件（假设是图像或其它文件）
        files = os.listdir(folder_path)

        # 打乱文件顺序
        random.shuffle(files)

        # 按照比例划分训练集和验证集
        train_count = int(len(files) * train_ratio)

        # 划分训练集和验证集文件
        train_files = files[:train_count]
        val_files = files[train_count:]

        # 创建对应的训练集和验证集对象文件夹
        train_folder = os.path.join(train_dir, folder)
        val_folder = os.path.join(val_dir, folder)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # 移动文件到训练集目录
        for file in train_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(train_folder, file)
            shutil.move(src_path, dst_path)

        # 移动文件到验证集目录
        for file in val_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(val_folder, file)
            shutil.move(src_path, dst_path)

        # 删除空的对象文件夹
        if not os.listdir(folder_path):  # 如果对象文件夹已空
            os.rmdir(folder_path)
            print(f"对象文件夹 {folder_path} 已删除")

        # 打印当前对象的划分情况
        print(f"对象 {folder} 分配到训练集: {len(train_files)} 文件, 验证集: {len(val_files)} 文件")

    print("数据集拆分完成")


if __name__ == '__main__':

    video_folder = '../data/train2'
    save_folder = "../data/temp/"
    train_dir = "../data/deepsort/train"  # 训练集目录
    val_dir = "../data/deepsort/test"   # 验证集目录

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    process_video(video_folder, save_folder)
    split_dataset(save_folder, train_dir, val_dir)


