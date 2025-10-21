!apt-get update
!apt-get install -y libnvinfer8 libnvinfer-dev libnvinfer-plugin8
!apt-get install -y graphsurgeon-tf uff-converter-tf
!nvidia-smi


!apt-get update
!apt-get install -y libnvinfer8 libnvinfer-plugin8 libnvinfer-dev

!pip install onnxruntime
!apt-get update -y
!apt-get install -y libnvinfer8 libnvinfer-dev libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8
!apt-get install -y tensorrt
!trtexec --version


!trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16


# option 2

!docker run --gpus all -it --rm nvcr.io/nvidia/tensorrt:23.12-py3


!apt-get update -y
!apt-get install -y --no-install-recommends \
    libnvinfer-dev libnvinfer-plugin-dev \
    libnvonnxparsers-dev libnvparsers-dev \
    libnvinfer-bin





  !find /usr/ -name trtexec

  !/usr/src/tensorrt/bin/trtexec --onnx=physical_ai_model.onnx --saveEngine=model_fp16.engine --fp16


  /bin/bash: line 1: /usr/src/tensorrt
  trtexec --onnx=physical_ai_model.onnx --saveEngine=model_fp16.engine --fp16

  !apt-get update -y
!apt-get install -y --no-install-recommends \
    libnvinfer-bin libnvinfer8 libnvinfer-plugin8 libnvinfer-dev \
    libnvonnxparsers8 libnvonnxparsers-dev libnvparsers8 libnvparsers-dev

!apt-get update -y
!apt-get install -y --no-install-recommends \
    libnvinfer-bin libnvinfer8 libnvinfer-plugin8 libnvinfer-dev \
    libnvonnxparsers8 libnvonnxparsers-dev libnvparsers8 libnvparsers-dev




!find /usr/ -type f -name trtexec



    !ls -lh model_fp16.engine

    !apt-get update -y
!apt-get install -y --no-install-recommends \
    libnvinfer8 libnvinfer-plugin8 libnvinfer-dev \
    libnvonnxparsers8 libnvparsers8 libnvinfer-bin


  !trtexec --version

  !docker run --gpus all -it --rm nvcr.io/nvidia/tensorrt:23.12-py3

  !apt-get update
!apt-get install -y chromium-browser
!pip install selenium webdriver-manager


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.google.com")
print("✅ Page title:", driver.title)
driver.quit()

!apt-get update
!apt-get install -y firefox
!pip install selenium

# from selenium import webdriver
from selenium.webdriver.firefox.options import Options

options = Options()
options.add_argument("--headless")
driver = webdriver.Firefox(options=options)
driver.get("https://www.python.org")
print("✅ Page title:", driver.title)
driver.quit()
# 
!apt-get install -y lynx w3m

!lynx -dump https://www.google.com

