#!/bin/bash

python ../datatools/video2frame.py
python ../datatools/mot2coco.py
python ../datatools/augscrop.py