# Use an official NVIDIA PyTorch runtime as a parent image
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set the working directory in the container to /app
# WORKDIR /app

# Add the current directory contents into the container
ADD /app .
# Install any needed packages
RUN apt-get update && \
    apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && \
    echo $CUDA_HOME && \
    git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    pip install -q -e GroundingDINO/ && \
    pip install uvicorn && \
    mkdir -p app/weights && \
    mkdir -p app/data && \
    pip uninstall -y supervision && \
    pip uninstall -y opencv-python && \
    pip install opencv-python==4.8.0.74 && \
    pip install -q supervision==0.6.0 && \
    pip install segment_anything && \
    pip install boto3 && \
    pip install botocore && \
    pip install fastapi && \
    pip install starlette && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P app/weights/ && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P app/weights/ && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P app/weights/ && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P app/weights/ && \
    wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg -P app/images/ 


# Copy over dummy modal package to replace the modal package used, to prevent modal errors
ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.8/site-packages/"
COPY app/dummy_modal.py /usr/local/lib/python3.8/site-packages/modal.py

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["uvicorn", "grounded_cutouts:app", "--host", "0.0.0.0", "--port", "80"]