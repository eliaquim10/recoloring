# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM continuumio/miniconda3

WORKDIR /opt
RUN mkdir -p /opt/code 
RUN cd /opt/code
COPY . code/

RUN apt update
RUN apt upgrade -y

RUN apt install unzip -y
RUN apt install gcc -y 

# RUN wget http://dev.ipol.im/~lraad/chromaGAN/data/chromagan_images.zip
RUN unzip code/chromagan_images.zip -d code/ 
RUN rm code/chromagan_images.zip
RUN mkdir -p code/sample_images
RUN mkdir -p code/chromagan_results

RUN conda env create -v --file code/ent.yml
# RUN conda create --name chroma python=3.7.6
RUN activate chroma

# RUN ln -f -s /usr/lib/x86_64-linux-gnu/libcuda.so.7.6.5 libcuda.so.1

RUN jupyter nbconvert DemoChromaGAN.ipynb --to python
RUN mv DemoChromaGAN.py /opt/code/DemoChromaGAN.py

# RUN pip install gdown
RUN gdown -O /tmp/my_model_colorization.h5 'https://drive.google.com/uc?id=12hyrk6A3bcpAFIBNW0sfeEevJ9daUTGr'
RUN mkdir -p /opt/code/MODEL/imagenet && mv /opt/code/my_model_colorization.h5 /opt/code/MODEL/imagenet/modelPretrained.h5

# pre-cache vgg16, não consegue acessa o cuda
# RUN /opt/conda/envs/chroma/bin/python -c "from keras import applications; applications.vgg16.VGG16(weights='imagenet', include_top=True)"

WORKDIR /opt/code/SOURCE
RUN sed -i "s/BATCH_SIZE = 10/BATCH_SIZE = 1/" config.py

ENTRYPOINT ["/opt/conda/envs/chroma/bin/python"]
# ENTRYPOINT ["/opt/conda/envs/chroma/bin/python", "ChromaGANPrint.py", "--no-concatenate"]