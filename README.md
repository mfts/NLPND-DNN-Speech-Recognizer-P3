[//]: # (Image References)

[image1]: ./images/pipeline.png "ASR Pipeline"
[image2]: ./images/select_kernel.png "select kernel"

# DNN Speech Recognizer

## Introduction

In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!  

![ASR Pipeline][image1]

We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate your models. Your algorithm will first convert any raw audio to feature representations that are commonly used for ASR. You will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, you will engage in your own investigations by creating and testing your own state-of-the-art models. Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations.

### Tasks

The tasks for this project are outlined in the `vui_notebook.ipynb` in three steps. Follow all the instructions, which include implementing code in `sample_models.py`, answering questions, and providing results. The following list is a summary of the required tasks.

#### Step 1 - Feature Extraction
- Execute all code cells to extract features from raw audio

#### Step 2 - Acoustic Model
- Implement the code for Models 1, 2, 3, and 4 in `sample_models.py`
- Train Models 0, 1, 2, 3, 4 in the notebook
- Execute the comparison code in the notebook
- Answer Question 1 in the notebook regarding the comparison
- Implement the code for the Final Model in `sample_models.py`
- Train the Final Model in the notebook
- Answer Question 2 in the notebook regarding your final model

#### Step 3 - Decoder
- Execute the prediction code in the notebook


## Installation

### Amazon Web Services (Cloud)

This project requires GPU acceleration to run efficiently. 

1. Follow the instructions to set up an AWS EC2 instance:
	- Launch Instance
	- Search for "Deep Learning" among the possible AMIs
	- Select a GPU instance (e.g. p2.xlarge)
	- Add a custom TCP rule (this is for Jupyter notebook): `{"Type": "Custom TCP Rule", "Protocol": "TCP", "Port Range": 8888, "Source": "Anywhere"}`
	- Launch _(set private/public key)_
	- Change permission of private key: `chmod 400 path/to/YourKeyName.pem`
	- Log in to instance: `ssh -i YourKeyName.pem ubuntu@X.X.X.X`
	- Update packages: `sudo apt-get update -y`
	- Install Libav Tools (to convert .flac to .wav): `sudo apt-get install -y libav-tools`

2. Clone repository
	```
	git clone https://github.com/mfts/NLPND-DNN-Speech-Recognizer-P3.git
	cd NLPND-DNN-Speech-Recognizer-P3
	```

3. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	```
	wget http://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzvf dev-clean.tar.gz
	wget http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	mv LibriSpeech data/LibriSpeech/ # move files to data folder
	mv data/flac_to_wav.sh data/LibriSpeech/flac_to_wav.sh # move flac_to_wav.sh inside the LibriSpeech folder
	./data/LibriSpeech/flac_to_wav.sh # run .flac to .wav file conversion
	```

4. Create JSON files corresponding to the train and validation datasets.
	```
	python create_desc_json.py data/LibriSpeech/dev-clean/ train_corpus.json
	python create_desc_json.py data/LibriSpeech/test-clean/ valid_corpus.json
	```

5. Start Jupyter:
	```
	jupyter notebook --ip=0.0.0.0 --no-browser
	```

6. Access the jupyter notebook from your web browser of choice
	- In the output of the terminal locate an URL that looks like: `http://0.0.0.0:8888/?token=...`. Copy the last portion of the URL starting with `:8888/?token=`
	- In the web browser enter `X.X.X.X:8888/?token=...` (where X.X.X.X is the IP address of your EC2 instance and everything starting with `:8888/?token=` is what you just copied

7. Open `vui_notebook.ipynb`

8. Change kernel to `Python [conda env:tensorflow2_p36]` _(last checked: 21.02.2020; subject to change on a different AMI)_


### Local Environment Setup

You should run this project with GPU acceleration for best performance.

1. Clone repository
	```
	git clone https://github.com/mfts/NLPND-DNN-Speech-Recognizer-P3.git
	cd NLPND-DNN-Speech-Recognizer-P3
	```

2. Create (and activate) a new environment with Python 3.6

	```
	conda create --name nlpnd python=3.5
	source activate nlpnd
	```

3. Install python packages
	```
	pip install -r requirements.txt
	```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the nlpnd environment
	```
	python -m ipykernel install --user --name nlpnd --display-name "nlpnd"
	```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```

6. Obtain the `libav` package.
	```
	brew install libav
	```

7. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	```
	wget http://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzvf dev-clean.tar.gz
	wget http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	mv LibriSpeech data/LibriSpeech/ # move files to data folder
	mv data/flac_to_wav.sh data/LibriSpeech/flac_to_wav.sh # move flac_to_wav.sh inside the LibriSpeech folder
	./data/LibriSpeech/flac_to_wav.sh # run .flac to .wav file conversion
	```

8. Create JSON files corresponding to the train and validation datasets.
	```
	python create_desc_json.py data/LibriSpeech/dev-clean/ train_corpus.json
	python create_desc_json.py data/LibriSpeech/test-clean/ valid_corpus.json
	```

9. Start Jupyter:
	```
	jupyter notebook
	```

10. Before running code, change the kernel to match the `nlpnd` environment by using the drop-down menu.  Then, follow the instructions in the notebook.

![select kernel][image2]

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__

---

## Special Thanks

We have borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
