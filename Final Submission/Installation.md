# Installiaton of Cuda Toolkit, CuDNN, Miniconda, and Jupyter Notebook

* The GPU supported simulator required Cuda and CuDNN as dependencies for JAX.
* Basic knowledge of Linux commandline usage is required. Used for file navigation and installation of tools. [This link](https://www.linuxjournal.com/content/linux-command-line-interface-introduction-guide) can be used for obtaining a basic overview for the setup of this project.
* Note for Windows Users: It is required to set up `WSL` as windows has compatability issues with JAX. After the installation of WSL, the commands for Linux can be inputted directly into the `WSL` commandline.
* Official instructions for WSL installation can be found through the [Official Microsoft Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install).
* Additional note to make sure to have the latest drivers for Nvidia installed on your computer, this installation guide will not cover this as it is based on your computer's manufacturer. Refer to instructions online for updating your specific computer's drivers.

## Cuda Toolkit Installation

* Cuda Toolkit provides a compiler for creating applications that run using the NVidia GPUs. Make sure to install a **11.x** version of the installation since CuDNN doesn't support **12.x** yet.
* The Cuda Toolkit installation page can be found at [Tookit Download Page](https://developer.nvidia.com/cuda-downloads?target_os=Linux).
  * Note that if you are using `WSL`, make sure to select the `WSL-Ubuntu` option, not the regular `Ubuntu` option.
* Select the latest version of the CUDA Toolkit and select the `runfile (local)` option.
* Then copy paste the commands into the command line.

## CuDNN Installation

* CuDNN is a collection of kernels that are used by JAX for optimization. The library is commonly used for Deep Learning but can also be applied here.
* Navigate to the [CuDNN download page](https://developer.nvidia.com/cudnn).
* Create a account or login to the CuDNN website to gain access to the CuDNN installation files.
* Agree to the terms and conditions and accept the Ethical AI agreement.
* Download the latest version of CuDNN for **Cuda 11.x**.
* Select `Local Installer For Ubuntu ... (Deb)` with the correct version for your system.
* In the commandline, after the download has finished, navigate to the folder that contains the file and type `sudo dpkg -i <file name>`. Replace `<file name>` with the filename.

## Miniconda and Jupyter

* Miniconda will be used to install packages and provide a virtual environment for python.
* Navigate to the [Miniconda Download Page](https://docs.conda.io/en/latest/miniconda.html) and install the correct version for linux.
* In the commandline, navigate to the folder that contains the downloaded file, usually the `Downloads/` folder.
* Run `bash <file name>` replacing `<file name>` with the filename of the downloaded file.
* Press enter to continue with the installation when the prompt shows up and follow through with the installation.
* After installation, type `conda activate <shell name>` where `<shell name>` is replaced by the shell that you are using. This can be found by typing the command `echo $SHELL`.
* Finally, close and reopen the command line.
* Now type `pip3 install jupyter notebook`. This will install the jupyter kernel for interacting with the code in the project.

## Jupyter Notebook Startup

* Jupyter Notebooks can be started by navigating the folder with the notebooks for the project and typing `jupyter notebook` in the commandline. This will open a webpage that will display the files in the current folder. Selecting a notebook file, will open the notebook in an interactive environment.
* It is recommended to get started with Jupyter Notebook through reading [this article](https://github.com/jupyter/notebook/blob/main/docs/source/examples/Notebook/Notebook%20Basics.ipynb) which gives an overview of the UI and how to open and edit notebooks.