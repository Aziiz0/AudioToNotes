# AudioToNotes

This repository contains the AudioToNotes project, which allows you to convert audio recordings into summarized notes.

## Prerequisites

- [Python 3.11.4](https://www.python.org/downloads/release/python-3114/)
- [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive)

## Installation

Follow these steps to set up the project:

1. Clone the repository:
    ```shell
    git clone https://github.com/YourUsername/AudioToNotes.git
    cd AudioToNotes
    ```

2. Create a virtual environment:
    ```shell
    python -m venv env
    ```

3. Activate the virtual environment:

    On Windows:
    ```shell
    .\env\Scripts\activate
    ```
    On Unix or MacOS:
    ```shell
    source env/bin/activate
    ```

4. Install the necessary Python packages:
    ```shell
    pip install -r requirements.txt
    ```

5. Install CUDA 11.7. You can download it from the [official website](https://developer.nvidia.com/cuda-downloads).

6. If necessary, update PyTorch to work with CUDA 11.7:

    On Windows:
    ```shell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    ```
    On Unix or MacOS, follow the instructions provided on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Usage

// Add instructions on how to run your project here.

## Contributing

// Add instructions on how to contribute to your project here.

## License

// Add information about your license here.
