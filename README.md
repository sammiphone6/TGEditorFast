# TGEditorFast

A fast implementation of TGEditor using Strategic Sampling.

## Setup

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd TGEditorFast
    ```
2. Create the conda environment:
    ```bash
    conda env create -f env.yml
    ```
3. Activate the environment:
    ```bash
    conda activate multignn
    ```

## Usage

Adjust the parameters at the top of `main.py` to select the dataset and the components of the pipeline to run. Then execute:

```bash
python main.py --mode YOUR_MODE --desc YOUR_DESCRIPTION
