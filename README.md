## MP3 to Text Converter with Dialogue Role and Sentiment Analysis

## Project description

This Python script is designed to automatically process MP3 audio files and convert them to text format (.txt). It performs the following actions:

1.  **Transcribe audio:** Uses OpenAI's Whisper model to recognize speech from audio files.
2.  **Dialog segmentation:** Applies inaSpeechSegmenter to divide audio into speech segments (male/female voice) and non-speech segments.
3.  **Role classification:** Uses RuBERT to determine the role of the speaker in a dialog (customer or salesperson).
4.  **Tone Analysis:** Applies TextBlob to analyze the tone of a text (polarity and subjectivity).
5.  **Saving results:** Transcribing, role and tonality analysis results are saved to text files (.txt) for each input MP3 file.

The script is designed to process dialog recordings, such as telephone conversations, for further analysis and text processing.

### Installation
### Pre-requisites
* **Python 3.7 or higher** (Python 3.8+ is recommended)
* **FFmpeg** must be installed and added to the system PATH variable (required to convert MP3 to WAV). Instructions for installing FFmpeg depend on your operating system.

### Installing the Python libraries

All dependencies must be installed before running the script. It is recommended to use a virtual environment (`venv`) to isolate the project dependencies.

1.**Create a virtual environment (optional, but recommended):**

    
```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    **Windows:**

        ```bash
        venv\Scripts\activate
        ```

    **Linux/macOS:**

        ```bash
        source venv/bin/activate
        ```
3.  ** Install the required libraries from the ``requirements.txt`` file:** ``bash source venv/bin/activate ``` 3.

    `````bash
    pip install -r requirements.txt
    ```
    (The `requirements.txt` file must be in the root folder of the project. Instructions for creating the `requirements.txt` file are below).