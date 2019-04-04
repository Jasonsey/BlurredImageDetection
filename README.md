# Blurred Image Detection

> detecting the blurred image
>

## Recognition effect

Testing on a data set of 214 blurred images and 241 clear images, the **F1 score** of  model's recognition is **0.81**. Here are partial outputs.

![](http://ww1.sinaimg.cn/large/006ztUIbgy1g0blps9yvwj30o00b614v.jpg)

## Project Structure

```shell
.
├── data								# data set folder
│   ├── input							# input data folder
│   │   └── License						# input data folder
│   │       ├── Test					# for test
│   │       │   ├── Bad_License
│   │       │   └── Good_License
│   │       └── Train					# for train
│   │           ├── Bad_License
│   │           └── Good_License
│   └── output							# output data folder
│       ├── cache						# cache data folder
│       ├── decision_tree				# decision tree model's model and log folder
│       │   └── models
│       ├── stacking					# stacking model's model and log folder
│       │   └── models
│       └── total_image					# CNN model's model and log folder
│           ├── log
│           └── models
├── docker-compose.yml					# docker compose configuration file
├── Dockerfile
├── Dockerfile.base						# the base dockerfile where the Dockerfile bases
├── interface.thrift					# thrift configuration file
├── Makefile							# pipline for docker container
├── requirements.txt					# dependency packages' information
├── docs
│   └── InterfaceDocument.md			# interface document
└── src									# source code
    ├── api								# api packages
    │   ├── __init__.py
    │   ├── cv2_api
    │   │   ├── __init__.py
    │   │   └── detection.py
    │   ├── decision_tree
    │   │   ├── __init__.py
    │   │   ├── detection.py
    │   │   └── train.py
    │   ├── stacking
    │   │   ├── __init__.py
    │   │   ├── detection.py
    │   │   └── train.py
    │   ├── thrift_api
    │   │   ├── __init__.py
    │   │   ├── interface				# thrift auto-generated folder
    │   └── total_image
    │       ├── __init__.py
    │       ├── detection.py
    │       ├── model.py
    │       └── train.py
    ├── dataset							# tools for reading data set
    │   ├── __init__.py
    │   ├── create_dataset.py
    │   └── read_dataset.py
    ├── deploy							# tools for deployment
    │   ├── __init__.py
    │   ├── detection.py
    │   ├── thrift_client.py
    │   └── thrift_server.py
    ├── utils							# comon tools
    │       ├── __init__.py
    │       ├── callbacks.py
    │       ├── params.py
    │       ├── server_tools.py
    │       └── tools.py
    ├── __main__.py
    ├── main.py							# main entry
    └── config.py						# global configuration file
```

## Project Introduction

The project consists of 3 models, CNN model, decision tree model and stacking model (another decision tree model to connect the two previous output and output the final result)

### CNN model

| 编号 | layers                 | filters | kernel_size | strides | padding | regularizers | activation |
| ---- | ---------------------- | ------- | ----------- | ------- | ------- | ------------ | ---------- |
| 1    | Conv2D                 | 64      | 3           | 1       | same    | l2(0.01)     | relu       |
| 2    | Conv2D                 | 64      | 3           | 1       | same    | None         | relu       |
| 3    | MaxPooling2D           | -       | 2           | 2       | -       | -            | -          |
| 4    | Conv2D                 | 64      | 3           | 1       | same    | None         | relu       |
| 5    | MaxPooling2D           | -       | 2           | 2       | -       | -            | -          |
| 6    | Conv2D                 | 64      | 3           | 1       | same    | None         | relu       |
| 7    | MaxPooling2D           | -       | 2           | 2       | -       | -            | -          |
| 8    | Conv2D                 | 64      | 3           | 1       | same    | None         | relu       |
| 9    | Conv2D                 | 1       | 3           | 1       | same    | None         | relu       |
| 10   | GlobalAveragePooling2D | -       | -           | -       | -       | -            | -          |
| 11   | Activation             | -       | -           | -       | -       | -            | sigmoid    |

### Decision tree model

The inputs of this model are image size, laplacian score and sober score. And the model's deep is set to 3

### Stacking model

The inputs of this model are the scores of CNN model and decision tree model. And the model's deep is set to 3 too.

## Rules for sample selection

Blurred image classification mainly serves the back-end OCR service, so the judgment of blurring is mainly based on the degree of text clarity. Here are the rules for sample selection.

1. Clear: The image of less than 2 blurred positions in the text blocks is clear.
2. Blur: More than 50% of the blurred positions in the text blocks are labeled as blurred.
3. If the left text blocks are clear (such as operator fields), then it is not included in the statistical range of text block. If most of the left text blocks is blurred, all the left text blocks are added to the statistical range.
4. Images with complex backgrounds or not business licenses will not be filtered as training samples.

## Q&A

### How to use the model simply

First, put blurred images into `data/input/License/Test/Bad License` and put clear images into `data/input/License/Test/Good License` . 

Second, run the following command.

```shell
$ cd src && python3.6 main.py test_stacking
```

Then the command line will show you the performance of the model in your data set.

### How to install python dependency packages

Just type the following command.

```shell
$ pip3.6 install -r requrements.txt
```

### How to start the blurred image detection service

First, edit the file `src/config.py` and set  your `THRIFT_HOST`, `THRIFT_PORT`, `THRIFT_NUM_WORKS`.

Second, run the following command.

```shell
$ cd src && python3.6 main.py server
```

### How to test the started service

First, put blurred images into `data/input/License/Test/Bad License` and put clear images into `data/input/License/Test/Good License` . 

Second, run the following command.

```shell
$ cd src && python3.6 main.py client
```

### How to Design Client Interface

See `docs` and you will find something you want.

### How to train the model with your data set

First, put blurred images into `data/input/License/Train/Bad License` and put clear images into `data/input/License/Train/Good License` . 

Second, run the following command.

```shell
$ cd src && python3.6 main.py train_all
```

### How to use docker

Fist, edit the file `src/config.py` and set  your `THRIFT_HOST`, `THRIFT_PORT`, `THRIFT_NUM_WORKS`

Second, edit the file `Makefile`  and delete the lines `docker tag ${base_name} ${remote_base}; docker push ${remote_base}`

Third, build the base docker image.

```shell
$ make build_base_image
```

Fourth, run the following command.

```shell
$ make up
```

Then, the service will start automatically.

### Other problems

Don't worry, I've commented on the code in detail. If you encounter any training problem, please feel free to contact me.