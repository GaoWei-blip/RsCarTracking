# 使用官方 Python 镜像
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# 复制应用代码到容器中
COPY . .

RUN cp start_jupyterlab.sh /

# 安装依赖
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y libgl1 libgl1-mesa-glx libglib2.0-0 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN #pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
#RUN pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple/
#RUN cd mmcv && pip install -e .
RUN pip install setuptools==59.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ &&  \
    pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple/ &&  \
    cd mmcv &&  \
    pip install -r requirements/optional.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ && pip install .


RUN cd cvtools && pip install .
RUN cd common/external && python setup.py build_ext --inplace

# 更新包列表并安装 zip
RUN apt-get update && apt-get install -y zip

