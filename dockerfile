# FROM python:3.12
# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04


EXPOSE 8501

WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip install --upgrade pip


# RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# RUN apt-get update
# RUN apt-get install -y nvidia-container-toolkit

COPY main.py /app
COPY cuda_test.py /app
COPY requirements.txt /app

COPY llama3.py /app
COPY Meta-Llama-3-8B-Instruct.Q4_1.gguf /app


RUN pip install streamlit

# RUN pip install  "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-nvcc -y --copy
RUN pip install cuda-python

# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1
# RUN pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

RUN FORCE_CMAKE=1
RUN LLAMA_CUDA=on
RUN CMAKE_ARGS="-DLLAMA_CUDA=on"
#
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

RUN pip install --upgrade --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --no-cache-dir
# RUN pip install --upgrade --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --no-cache-dir
# --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/libcuda.so.1

ENTRYPOINT [ "streamlit", "run"]
CMD ["main.py"]