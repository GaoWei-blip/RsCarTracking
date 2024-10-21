# 遥感视频目标追踪

- DSFNet+SORT
- 包含：可变性卷积DCNv3和DeepSORT

```shell
# 环境
conda create -n track python=3.8
pip install -r requirements.txt
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
cd cvtools
pip install -e .

# nms
cd common/external
python setup.py build_ext --inplace

# DCNv3
cd model/DCNv3
python setup.py install

```

```shell
# 数据处理
process.sh

# 训练
train.sh

# 预测并追踪
test.sh


# nohup python train.py > output.log 2>&1 &
# tail -f output.log
# pkill -f train.py
```
