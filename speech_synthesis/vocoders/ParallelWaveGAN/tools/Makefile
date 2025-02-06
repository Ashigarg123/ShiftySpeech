PYTHON:= python3.8  # Set the Python version (not required if the environment is already set)
CUDA_VERSION:= 11.3  # Set the CUDA version
PYTORCH_VERSION:= 1.12.1  # Set the PyTorch version
ENV_NAME:= train_vocoders  # Name of your existing Conda environment
DOT:= .

.PHONY: all clean show_variables

all: show_variables pytorch.done parallel_wavegan.done

show_variables:
	@echo PYTHON=$(PYTHON)
	@echo CUDA_VERSION=$(CUDA_VERSION)
	@echo PYTORCH_VERSION=$(PYTORCH_VERSION)
	@echo ENV_NAME=$(ENV_NAME)

# Step 1: Install PyTorch
pytorch.done:
	conda install -n $(ENV_NAME) pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
	touch pytorch.done

# Step 2: Install Parallel WaveGAN
parallel_wavegan.done: pytorch.done
	conda run -n $(ENV_NAME) pip install -e .
	conda run -n $(ENV_NAME) pip install -e .[test]
	touch parallel_wavegan.done

# (Optional) Install Apex
apex.done: pytorch.done
	git clone https://github.com/NVIDIA/apex.git
	conda run -n $(ENV_NAME) pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
	touch apex.done

# Clean up only generated files
clean:
	rm -fr apex *.done
	find -iname "*.pyc" -delete
