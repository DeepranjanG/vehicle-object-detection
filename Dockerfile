FROM continuumio/miniconda3
COPY . /vehicle
WORKDIR /vehicle
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
RUN conda install -c conda-forge pycocotools
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .
CMD ["python","app.py"]