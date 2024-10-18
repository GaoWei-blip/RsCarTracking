# _*_ coding:utf-8 _*_

import shutil
import cv2
import os
import sys
base_dir = os.getcwd()
sys.path[0] = base_dir


def video_get_img(videoPath, svPath, max_frames=300):
    # 读取视频
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    print(f"FPS: {fps} for video: {videoPath}")
    while True:
        # 按帧读取视频，返回值ret是布尔型，正确读取则返回True
        flag, frame = cap.read()
        if not flag:
            break

        numFrame += 1
        # 拼接图片保存路径
        newPath = os.path.join(svPath, str(numFrame).zfill(6) + ".bmp")
        # 将图片按照设置格式，保存到文件
        cv2.imencode('.bmp', frame)[1].tofile(newPath)

        if numFrame >= max_frames:
            # 只保存指定数量的图片
            break

    cap.release()  # 释放读取画面状态


def copy_video(src_dir, dst_dir):
    # 遍历源目录中的所有文件
    for file_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        # 检查是否是文件
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)  # 使用 copy2 保持文件元数据
            print(f"已复制: {src_file} 到 {dst_file}")


if __name__ == '__main__':

    SPLITS = ['train', 'val']

    if not os.path.exists('/work/train'):
        os.mkdir('/work/train')
    copy_video('/data/train1', '/work/train')
    copy_video('/data/train2', '/work/train')

    # 从训练集中提取验证集
    files = os.listdir('/data/train2')
    csv_file = ''
    val_count = 0
    val_num = 2
    for file in files:
        if file.endswith('.avi'):
            base_name = file[:-4]  # 去掉扩展名后的文件名
            csv_file = f"{base_name}-gt.csv"
            if not os.path.exists('/work/train2Val'):
                os.mkdir('/work/train2Val')
            shutil.copy(os.path.join('/data/train2', csv_file), '/work/train2Val')
            shutil.copy(os.path.join('/data/train2', f"{base_name}.avi"), '/work/train2Val')
            val_count += 1
            if val_count >= val_num:
                break

    for split in SPLITS:
        if split == 'train':
            video_folder = '/work/train'
            save_folder = r"/work/RsCarData/images/train"
        elif split == 'val':
            video_folder = '/work/train2Val'
            save_folder = r"/work/RsCarData/images/val"

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

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
                        'avi': file,
                        'csv': csv_name
                    }

        for base_name, pair in file_pairs.items():
            # 获取视频文件和gt文件的名称
            video_name = pair['avi']
            csv_file_path = pair['csv']

            # 获取视频文件的完整路径
            videopath = os.path.join(video_folder, video_name)
            if not os.path.isfile(videopath):
                continue

            # 获取视频文件的名称（不带扩展名）
            video_name = os.path.splitext(video_name)[0]

            # 为每个视频文件创建一个单独的文件夹
            svpath = os.path.join(save_folder, video_name)
            if not os.path.exists(svpath):
                os.mkdir(svpath)
            svpath2 = os.path.join(svpath, "img")
            if not os.path.exists(svpath2):
                os.mkdir(svpath2)
            svpath3 = os.path.join(svpath, "gt")
            if not os.path.exists(svpath3):
                os.mkdir(svpath3)

            # 移动gt文件
            destination_file = os.path.join(svpath3, csv_file_path)
            shutil.copy(os.path.join(video_folder, csv_file_path), destination_file)
            # 处理视频并保存图片
            video_get_img(videopath, svpath2)
            print(f"{video_name} processed")

        print("All videos processed")
