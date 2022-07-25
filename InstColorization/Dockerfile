FROM continuumio/miniconda3

WORKDIR /opt/notebooks

# Create the environment:
#clona o repositorio para dentro do container docker
RUN git clone https://github.com/ericsujw/InstColorization .

# instala pacotes do python e para o sistema
RUN apt update -y 
RUN apt upgrade -y 
RUN conda env create -f env.yml
RUN activate instacolorization
RUN conda run pip install IProgress
RUN conda run pip install cython pyyaml==5.1
RUN conda install -n instacolorization ipykernel --update-deps --force-reinstall
RUN apt install gcc -y
RUN apt install unzip -y
RUN apt install zip -y
# RUN conda run pip install -U torch==1.6 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
# RUN conda run pip install -U torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN conda run /opt/conda/envs/instacolorization/bin/python -m pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html -V  
RUN conda run /opt/conda/envs/instacolorization/bin/python -m pip install ipywidgets==7.4.2
RUN conda run /opt/conda/envs/instacolorization/bin/python -m pip install google-colab>=1.0.0

RUN conda run /opt/conda/envs/instacolorization/bin/python -m pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN conda run /opt/conda/envs/instacolorization/bin/python -m pip install dominate==2.4.0
RUN conda run /opt/conda/envs/instacolorization/bin/python -m pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html

ENTRYPOINT ["python"]
# /root/.torch/fvcore_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl
# ["python", "download.py"]
# docker run -i --name recolor --gpus all continuumio/miniconda3
# docker run -i --name color --gpus all colorization
# docker cp checkpoints/ color:/opt/notebooks/
# docker build -t colorization .

# docker run -i --name et --gpus all entity_relation
# docker system prune