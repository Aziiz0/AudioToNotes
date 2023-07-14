# AudioToNotes

This repository contains the AudioToNotes project, which allows you to convert audio recordings into summarized notes.

## Prerequisites

- [Python 3.11.4](https://www.python.org/downloads/release/python-3114/)
- [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive)

## Setting Up Environment Variables

This project requires three environment variables: `NOTION_DATABASE_ID`, `NOTION_TOKEN`, and `OpenAi_API_Key`. Here's how to set them up:

1. `NOTION_DATABASE_ID`: This is the ID of your Notion database. To find it, open your Notion page. The URL will look something like this: `https://www.notion.so/<long_hash_1>?v=<long_hash_2>`. The `<long_hash_1>` part is your database ID.

2. `NOTION_TOKEN`: This is your Notion Secret Key. To get it, go to the [My Integrations](https://www.notion.so/my-integrations) page on the Notion website. If you haven't created an integration yet, you can do so on this page. Once you have an integration, you can find your Secret Key on its details page.

3. `OpenAi_API_Key`: This is your OpenAI API Key. To get it, go to the [API Keys](https://platform.openai.com/account/api-keys) page on the OpenAI platform. You'll need to create an account if you don't have one already. Once you're logged in, you can create a new API key.

To set these variables locally:

- On Windows, use the command `setx <variable_name> <value>` in the command prompt.

- On Unix or MacOS, add `export <variable_name>=<value>` to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`).

Replace `<variable_name>` with the name of the variable (e.g., `NOTION_DATABASE_ID`) and `<value>` with the corresponding value you found in the steps above.

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

5. Install the necessary Python packages:
    ```shell
    pip install -r requirements.txt
    ```
    
6. Download the necessary NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    ```
    This command should be run in the Python environment where you installed the requirements.

If necessary, update PyTorch to work with CUDA 11.7:
    On Windows:
    ```shell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    ```
    On Unix or MacOS, follow the instructions provided on the [PyTorch website](https://pytorch.org/get-started/locally/).
