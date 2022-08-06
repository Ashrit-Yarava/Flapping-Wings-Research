# Environment Setup For Development In Python

## Table Of Contents

1. Windows Subsystem For Linux (WSL)
2. Installing Conda
3. Creating the Conda Environment
4. Installing Other Dependencies
5. Installing Jax
6. VS Code For Development
7. Jupyter Notebook
8. Other Notes
9. Resetting/Deleting the Conda Environment

## Windows Subsystem For Linux (WSL)

> Windows users will need to install the Windows Subsystem for Linux which will install and underlying linux kernel that can be used by JAX. Although it is possible to use JAX on windows natively, it is minimally tested and a release build of JAX is not yet available so building from source is the only way to install it.

1. Open PowerShell as administrator.

2. Run the command below. It installs WSL onto the system. Follow through with the instructions outputted by the command, it will ask you to set up a username/password.

   ```powershell
   wsl --install
   ```

3. After installation has completed, there will be a new app called `Ubuntu`. By opening the app, Windows connects you to the linux VM running on the machine.

4. Commands following this will be expected to be run in WSL unless explicitly stated otherwise.

5. When referring to opening the terminal in later parts of the manual, open the `Ubuntu` App.

## Setting Up Conda

> Although not necessary, Conda makes it easy to manage multiple installations of python and isolates these instances from the rest of the system. WSL users will install Conda on the linux system and use it from there.

1. Open the terminal.

2. Run the following commands in order in the newly opened terminal.

   ```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   ./Miniconda3-latest-Linux-x86_64.sh
   ```

3. Follow the instructions on screen to install miniconda.

   1. Installing miniconda is better than installing the standard version of anaconda in this scenario as it does not provide preinstalled files which include a lot of unused packages for this project.

4. Run the following command to initialize conda and then, close and reopen the terminal. Replace shellname with the shell that you are currently using. This can be identified with `which $SHELL` which will output the path of the shell you are using. Take the word after the last backslash (`/`).

   ```
   conda init SHELLNAME
   ```

5. Upon reopening the terminal, there must be a `(base)` somewhere around the PROMPT. This indicates that conda is working properly and that the `base` environment has been selected.

## Creating the Conda Environment

> The conda environment created below will act as a container for all the packages that are going to be used. Make sure to activate it before running anything in the terminal regarding this project.

1. Run the following command to create the new environment. Confirm when the prompt asks you whether you want to go through with the installation.

   ```
   conda create -n wings python=3.9
   ```

2. The environment can then be activated with the following command, make sure to run it everytime you are working on this project. The prompt where `(base)` appeared will have changed to `(wings)`

   ```
   conda activate wings
   ```

3. Update the pip package manager. Ensure that it is the right one.

   ```
   conda install --force pip
   which pip3
   ```

   The last command should show you a path with the miniconda environment. A sample output: `/opt/homebrew/Caskroom/mambaforge/base/envs/wings/bin/pip3`. The `base/envs/wings` part is the most important.

4. Now install the requirements found in the `requirements.txt` file

   1. First clone the repository.

      ```
      git clone https://github.com/Ashrit-Yarava/Flapping-Wings-Research.git
      cd Flapping-Wings-Research/
      ```

   2. Install the pip requirements.

      ```
      pip3 install -r requirements.txt
      ```

5. This will install all the other dependencies for the project as well as jupyter notebook.

## Installing Other Dependencies (GPU ONLY)

> JAX relies on the CUDA toolkit as well as the CuDNN library. The CUDA toolkit is used for interacting with the NVIDIA GPUs while the CuDNN library is used for compiling functions with XLA to be capable of running on the GPU.

1. Install the NVIDIA toolkit from the NVIDIA cuda toolkit link listed below. Follow the instructions for the WSL-UBUNTU option.

<<<<<<< HEAD:Resources/Conda Environment Setup.md
   1. Note that it is required to install the Game Ready Nvidia Driver on the windows operating system as well. This allows for communication between the GPU and the linux subsystem.

=======
   ```
   
   ```
   
   If a password prompt is requested, enter the password for the Linux account.
   
>>>>>>> main:Conda Environment Setup.md
2. Ensure that the CUDA toolkit is installed by running:

   ```
   nvcc --version
   ```

   It should output something similar to although the version numbers will most likely be different:

   ```
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2019 NVIDIA Corporation
   Built on Sun_Jul_28_19:07:16_PDT_2019
   Cuda compilation tools, release 10.1, V10.1.243
   ```

3. Installing the CuDNN library requires registering for the NVIDIA. Click `Download cuDNN` on the [NVIDIA website](https://developer.nvidia.com/cudnn). Create an account or login and get the `.deb` file.

4. After downloading the file, install it through apt. Navigating to the file through the terminal can be done via `cd (dirname)`. Note that the windows filesystem is mounted to the linux system through `/mnt/c` so entering

   ```
   cd /mnt/c/
   ```

   with navigate you to the windows system.

5. You can check the list of files and folders in the current directory with `ls` command.

6. After finding the cuDNN `.deb` file, install it via the command below.

   ```
   sudo apt install ./CUDNNFILE
   ```

   where `CUDNNFILE` is the filename of the `.deb` file.

## Installing JAX

> There are two possible choices for installation, with or without GPU support. GPU support is only available with an NVIDIA GPU however, with support for AMD GPUs being preliminary. See the [issue](https://github.com/google/jax/issues/2012) for more details on installation and current bugs.

#### CPU Only

1. Run the following command below to instlal the package.

   ```
   pip3 install jax jaxlib
   ```

2. Check that the JAX installation works by running the following command below.

   ```
   python3 -c 'import jax;print(jax.devices())'
   ```

   You should see a `CpuDevice` listed after the command runs. Running this on an M1 MAC will result in a warning about minimal testing which can be safely ignored. If there are any other errors, contact me through CANVAS.

## GPU Support

1. When running on the GPU, JAX will automatically choose the GPU for hardware optimizations.

2. Run the following command below to install JAX on gpu.

   ```
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
   ```

3. Run the following command to ensure that JAX is working properly.

   ```
   python3 -c 'import jax;print(jax.devices())'
   ```

   You should see a `GpuDevice` or multiple GPUs if your system has multiple GPUs.

## VS Code For Development

> Visual Studio Code is a product provided by microsoft for development. It is especially good for python due to features such as autocompletion, syntax highlighting, linting, auto imports, and support for jupyter notebooks in the editor itself.

1. Install and the open VS Code. Installation can be done via the link below:

   https://code.visualstudio.com

2. Navigate to the extensions tab and search for `Python`. Install the first extension listed and open the `igVortex/Python` folder through VS Code by selecting `File` and `Open...`

3. Open the `igVortex.py` file. This will indicate to VS Code that this is a Python project.

4. Click on the numbers below and select the conda `wings` environment.

5. This will automatically be used for later projects. But make sure to check whenever you open VS Code that the correct python environment is being used.

### WSL Users

> VS Code allows for interfacing with the WSL environment.

1. Install the `Remote - WSL` Extension through the extensions tab.

2. VS Code can then be launched through the `Ubuntu` app by navigating to the directory you want to open in VS Code and running:

   ```
   code .
   ```

## Jupyter Notebook

> Jupyter Notebooks are interactive python environments that allow for code editing in an isolated environment. Jupyter Notebook files are indicated by the `.ipynb` file. Opening this file in VS Code will automatically set up the Notebook.

1. Open the sample jupyter notebook in the folder.

   ![Screen Shot 2022-06-16 at 11.36.04 AM](Images/Screen Shot 2022-06-16 at 11.36.04 AM.png)

2.  Select the kernel such that it shows the above kernel. Now you can select the ▷ button to run each box. The boxes can be run in any order. The environment can also be restarted by clicking the restart button or stop the current box's execution with the `Interrupt` button.

## Running the Project

1. Open VS Code in the `igVortex/Python` folder and type the command below to run the program.

   ```
   python3 igVortex.py
   ```

   You can also click the run button while in the `igVortex.py` folder is run the file.

2. The `src` folder contains the other functions that are similar to the MATLAB code. It is necessary to include them in the `src` folder as Python treats the src folder as a package and allows for imports within the project.

## Resetting/Deleting the Conda Environment

> If there is an issue with the conda environment, it can be reset by deleting the environment and following through with the environment setup stated previously. Although it is not necessary to reinstall the CUDA toolkit/CuDNN again.

1. Remove the `wings` environment.

   ```
   conda remove -n wings --all
   ```

   Enter `y` to confirm.
