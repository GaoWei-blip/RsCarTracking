#!/bin/bash

pip install jupyterlab

jupyter-lab --port 8888 --allow-root --NotebookApp.ip=0.0.0.0 --NotebookApp.token=''