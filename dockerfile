# FROM python:3.12
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

EXPOSE 8501

WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip install --upgrade pip

COPY main.py /app
COPY cuda_test.py /app
COPY requirements.txt /app

COPY llama3.py /app
COPY Meta-Llama-3-8B-Instruct.Q4_1.gguf /app


RUN pip install streamlit

# RUN pip install  "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc -y --copy
RUN pip install cuda-python

RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1
RUN pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir


ENTRYPOINT [ "streamlit", "run"]
CMD ["main.py"]