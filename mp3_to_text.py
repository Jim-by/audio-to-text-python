import os
import subprocess
import torch
import warnings
from pathlib import Path
from typing import Tuple, Dict, List

import whisper
from inaSpeechSegmenter import Segmenter
from pydub import AudioSegment
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Disabling warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
BASE_DIR = Path(__file__).parent.resolve()  # Project root folder
INPUT_MP3_DIR = BASE_DIR / "input_mp3"      # Input MP3s folder
OUTPUT_TXT_DIR = BASE_DIR / "output_txt"    # Result folder

# Create folders if there are none
INPUT_MP3_DIR.mkdir(exist_ok=True)
OUTPUT_TXT_DIR.mkdir(exist_ok=True)

# --- Loading models ---
def load_models() -> Tuple[whisper.Whisper, AutoTokenizer, AutoModelForSequenceClassification]:
    """Loading Whisper and RuBERT models with error handling."""
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("medium")

        print("Loading RuBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased",
            num_labels=2
        )
        return model, tokenizer, bert_model
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

whisper_model, rubert_tokenizer, rubert_model = load_models()

# --- Helper functions ---
def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """Convert MP3 to WAV (16 kHz, mono) using ffmpeg."""
    try:
        cmd = [
            'ffmpeg',
            '-i', str(mp3_path),
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            str(wav_path),
            '-y'
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion error: {e}")
        return False

def determine_role_with_rubert(text: str) -> str:
    """Classify speaker role (client/seller) using RuBERT."""
    try:
        inputs = rubert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = rubert_model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()
        return ["client", "seller"][predicted_label]
    except Exception as e:
        print(f"Classification error: {e}")
        return "unknown"

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze text sentiment using TextBlob."""
    try:
        analysis = TextBlob(text)
        return {
            "polarity": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity
        }
    except Exception:
        return {"polarity": 0.0, "subjectivity": 0.0}

# --- Main processing ---
def process_audio_file(mp3_path: Path, output_txt_path: Path) -> bool:
    """Process single MP3 file through full pipeline."""
    # Step 1: Convert to WAV
    temp_wav = BASE_DIR / "temp.wav"
    if not convert_mp3_to_wav(mp3_path, temp_wav):
        return False

    # Load audio for segmentation
    try:
        audio = AudioSegment.from_wav(str(temp_wav))
    except Exception as e:
        print(f"Error loading audio: {e}")
        temp_wav.unlink(missing_ok=True)
        return False

    # Step 2: Voice activity detection
    print("Performing audio segmentation...")
    try:
        segmenter = Segmenter()
        segments = segmenter(str(temp_wav))
    except Exception as e:
        print(f"Segmentation error: {e}")
        temp_wav.unlink(missing_ok=True)
        return False

    final_dialogue = []

    # Step 3: Process each segment
    for seg in segments:
        label, seg_start, seg_end = seg
        if label not in {"male", "female"}:
            continue  # Skip non-speech segments

        # Convert time to milliseconds
        seg_start_ms = int(seg_start * 1000)
        seg_end_ms = int(seg_end * 1000)

        # Extract audio segment
        segment_audio = audio[seg_start_ms:seg_end_ms]
        segment_wav = BASE_DIR / "segment_temp.wav"
        segment_audio.export(str(segment_wav), format="wav")

        # Step 4: Transcribe with Whisper
        try:
            transcription_result = whisper_model.transcribe(
                str(segment_wav),
                language="ru",
                fp16=torch.cuda.is_available()
            )
            segment_text = transcription_result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            segment_text = ""

        # Clean up segment file
        segment_wav.unlink(missing_ok=True)

        if not segment_text:
            continue

        # Step 5: Analyze segment
        role = determine_role_with_rubert(segment_text)
        sentiment = analyze_sentiment(segment_text)

        entry = (
            f"{role.capitalize()} ({label}): {segment_text}\n"
            f"Sentiment: Polarity={sentiment['polarity']:.2f}, "
            f"Subjectivity={sentiment['subjectivity']:.2f}\n"
            f"Time: {seg_start:.1f}-{seg_end:.1f}s\n"
        )
        final_dialogue.append(entry)

    # Step 6: Save results
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(final_dialogue))
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

    # Clean up
    temp_wav.unlink(missing_ok=True)
    return True

# --- Entry point ---
def main():
    print(f"Looking for MP3 files in {INPUT_MP3_DIR}...")
    mp3_files = list(INPUT_MP3_DIR.glob("*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in {INPUT_MP3_DIR}")
        print(f"Please place your MP3 files in the '{INPUT_MP3_DIR.name}' folder")
        return

    print(f"Found {len(mp3_files)} files to process")
    
    for mp3_file in mp3_files:
        print(f"\nProcessing: {mp3_file.name}")
        output_txt = OUTPUT_TXT_DIR / f"{mp3_file.stem}.txt"

        success = process_audio_file(mp3_file, output_txt)
        if success:
            print(f"Success! Results saved to {output_txt}")
        else:
            print(f"Failed to process {mp3_file.name}")

    print("\nProcessing complete. Results saved in:", OUTPUT_TXT_DIR)

if __name__ == "__main__":
    main()