## YOLOv7

    Train YOLOv7 example: https://youtu.be/-QWxJ0j9EY8


# Download torch and torchvision

    - Links for torch:
        https://download.pytorch.org/whl/torch/

    - Links for torchvision:
        https://download.pytorch.org/whl/torchvision/

    - example( PYTHON == 3.9, NVIDIA CUDA == 10.2, Windows X64 ):
        torch download : torch-1.10.2+cu102-cp39-cp39-win_amd64.whl
        torchvision download : torchvision-0.11.3+cu102-cp39-cp39-win_amd64.whl


## Install Tesseract-OCR

    - Installer can be downloaded here:
        https://github.com/UB-Mannheim/tesseract/wiki

    - Install language pack
        ./tessdata -> (the default path )C:\Program Files\Tesseract-OCR\tessdata

    - Environment configuration
        Add [TESSDATA_PREFIX] environment variable
        ．C:\Program Files\Tesseract-OCR\tessdata
        Environment variable PATH added
        ．C:\Program Files\Tesseract-OCR\tessdata
        ．C:\Program Files\Tesseract-OCR

# Install required Python packages:

    $ pip install -r requirements.txt

    $ pip install { torch download file path }

    $ pip install { torchvision download file path }


# PyTorch版本1.10，安裝完tensorboard後運行代碼時出現錯誤：AttributeError: module 'distutils' has no attribute 'version'。

    $ pip uninstall setuptools

    $ pip install setuptools==59.5.0



## Terminal 1 ( Redis位置 )

    redis-server.exe  redis.windows.conf

    如果出現 # Creating Server TCP listening socket 127.0.0.1:6379: bind: No error
    $ redis-cli.exe
    127.0.0.1:6379> shutdown
    not connected> exit
    redis-server.exe  redis.windows.conf

## Terminal 2

    $ celery -A app.celery.tasks worker --loglevel=INFO -P eventlet

## Terminal 3

    $ python manage.py runserver
## YOLOv7

    Train YOLOv7 example: https://youtu.be/-QWxJ0j9EY8


# Download torch and torchvision

    - Links for torch:
        https://download.pytorch.org/whl/torch/

    - Links for torchvision:
        https://download.pytorch.org/whl/torchvision/

    - example( PYTHON == 3.9, NVIDIA CUDA == 10.2, Windows X64 ):
        torch download : torch-1.10.2+cu102-cp39-cp39-win_amd64.whl
        torchvision download : torchvision-0.11.3+cu102-cp39-cp39-win_amd64.whl


## Install Tesseract-OCR

    - Installer can be downloaded here:
        https://github.com/UB-Mannheim/tesseract/wiki

    - Install language pack
        ./tessdata -> (the default path )C:\Program Files\Tesseract-OCR\tessdata

    - Environment configuration
        Add [TESSDATA_PREFIX] environment variable
        ．C:\Program Files\Tesseract-OCR\tessdata
        Environment variable PATH added
        ．C:\Program Files\Tesseract-OCR\tessdata
        ．C:\Program Files\Tesseract-OCR

# Install required Python packages:

    $ pip install -r requirements.txt

    $ pip install { torch download file path }

    $ pip install { torchvision download file path }


# PyTorch版本1.10，安裝完tensorboard後運行代碼時出現錯誤：AttributeError: module 'distutils' has no attribute 'version'。

    $ pip uninstall setuptools

    $ pip install setuptools==59.5.0



## Terminal 1 ( Redis位置 )

    redis-server.exe  redis.windows.conf

    如果出現 # Creating Server TCP listening socket 127.0.0.1:6379: bind: No error
    $ redis-cli.exe
    127.0.0.1:6379> shutdown
    not connected> exit
    redis-server.exe  redis.windows.conf

## Terminal 2

    $ celery -A app.celery.tasks worker --loglevel=INFO -P eventlet

## Terminal 3

    $ python manage.py runserver
