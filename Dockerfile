FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt-get update -y

# Timezone settings for OpenCV
RUN ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y python3-opencv

RUN pip install ray[default]
RUN pip install ray[tune] wandb matplotlib opencv-python pandas pillow progressbar2 tabulate
RUN pip install --no-deps efficient-det
