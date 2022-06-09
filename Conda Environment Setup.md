# Conda Package Manager and Package Setup

> Note that this is entirely for setup on personal use computers, Google Colab provides all these packages preinstalled.
>
> Using Windows Subsystem for Linux to setup this environment is preferred compared to running it natively on Windows as JAX is more mature on the linux subsystem and will be less prone to bugs.

1. Firstly, it is important to install a version of the conda package management environment. This will allow for a unified package versioning such that there are no discrepancies between the code. Installer webpage/information: https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links

2. Follow through with the installation instructions provided by your selected installation method and ensure that conda is found in your system $PATH.

3. Now using your preferred terminal/command prompt, create a new conda environment that will be used to store the packages for this project.

   ```bash
   conda create -n wings python=3.9
   ```

4. The conda environment can then be activated later on with:

   ```bash
   conda activate wings
   pip3 install --upgrade pip
   ```

5. Now there are 2 possible approaches to installation, with GPU support or without:

## CPU only

1. Run the command below in the root of the repository after cloning it from GitHub.

   ```bash
   pip3 install -r requirements-cpu.txt
   ```

## GPU support

1. CUDA and CuDNN must be installed on the computer. Refer to the the NVIDIA page for information of the installation of these tools. Make sure to install the latest versions of both of these tools.

2. Install JAX built with CUDA support.

   ```
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
   ```

3. Install other dependences. Must be run in the root of the repository.

   ```
   pip3 install -r requirements-gpu.txt
   ```

#### Make sure to select the wings conda interpreter in your preferred editor before editing.

## Running the Code

Either using the editor or the terminal, run the `igVortex.py` file.

```
python3 igVortex.py
```

Must be done in the `igVortex/Python` directory.

