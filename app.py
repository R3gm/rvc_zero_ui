import os
import subprocess
import gradio as gr
import spaces
from infer_rvc_python import BaseLoader
import random
import logging
import time
import soundfile as sf
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
import zipfile
import edge_tts
import asyncio
import librosa
import traceback
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import urllib.request
import urllib.parse
import urllib.error
import shutil
import threading
import argparse
import sys
import torch
from picklescan.scanner import scan_file_path
import hashlib
import json
import uuid
import tempfile
import sqlite3
import pandas as pd
import unicodedata
import re
import ast
from huggingface_hub import HfApi

ALLOWED_DOMAINS = {"huggingface.co", "hf.co"}
MODEL_CACHE = {}
URL_CACHE = {}
LIKES_CACHE = {}

parser = argparse.ArgumentParser(description="Run the app with optional sharing")
parser.add_argument(
    '--share',
    action='store_true',
    help='Enable sharing mode'
)
parser.add_argument(
    '--theme',
    type=str,
    default="default_theme",
    help='Set the theme (default: default_theme)'
)
args = parser.parse_args()

IS_COLAB = True if ('google.colab' in sys.modules or args.share) else False
IS_ZERO_GPU = os.getenv("SPACES_ZERO_GPU")

if args.theme == "default_theme":
    selected_theme = gr.themes.Monochrome()
    IS_R3_THEME = True
else:
    selected_theme = args.theme
    IS_R3_THEME = False

disable_progress_bars()
logging.getLogger("infer_rvc_python").setLevel(logging.WARNING)

rmvpe_path = hf_hub_download(
    repo_id="r3gm/sonitranslate_voice_models",
    filename="rmvpe.pt",
)
converter = BaseLoader(only_cpu=False, hubert_path="r3gm/hubert_base", rmvpe_path=rmvpe_path, preload_models=True)


def parse_hf_url(url: str):
    url = urllib.parse.unquote(url)
    url = url.split("?")[0].strip()
    url = url.replace("https://huggingface.co/", "").replace("https://hf.co/", "")
    
    parts = url.split("/")
    repo_id = f"{parts[0]}/{parts[1]}"
    
    if len(parts) >= 5 and parts[2] in ["resolve", "blob"]:
        revision = parts[3]
        filename = "/".join(parts[4:])
    else:
        revision = "main"
        filename = "/".join(parts[2:])
        
    return repo_id, filename, revision


def hf_download_file(url: str, directory: str):
    repo_id, filename, revision = parse_hf_url(url)
    repo_id, filename, revision = parse_hf_url(url)
    
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=directory,
        local_dir_use_symlinks=False
    )
    
    base_filename = os.path.basename(filename)
    target_path = os.path.join(directory, base_filename)
    
    if os.path.abspath(downloaded_path) != os.path.abspath(target_path):
        subfolder_dir = os.path.dirname(downloaded_path)
        shutil.move(downloaded_path, target_path)
        
        # Remove empty subfolder(s) left behind
        try:
            os.removedirs(subfolder_dir)
        except OSError:
            pass

    return target_path


test_model_urls = [
    "https://huggingface.co/r3gm/villager/resolve/main/model.pth",
    "https://huggingface.co/r3gm/villager/resolve/main/model.index"
]
test_names = ["model.pth", "model.index"]

for url, filename in zip(test_model_urls, test_names):
    if os.path.exists(filename):
        continue
    try:
        hf_download_file(url, directory=".")
        if not os.path.isfile(filename):
            raise FileNotFoundError
    except Exception:
        with open(filename, "wb") as f:
            pass

TITLE = "<center><strong><font size='7'>RVC⚡ZERO</font></strong></center>"
DESCRIPTION = "This demo is provided for educational and research purposes only. The authors and contributors of this project do not endorse or encourage any misuse or unethical use of this software. Any use of this software for purposes other than those intended is solely at the user's own risk. The authors and contributors shall not be held responsible for any damages or liabilities arising from the use of this demo inappropriately." if IS_ZERO_GPU else ""
TEST_MODEL = ", ".join(test_model_urls)
ZIP_EXAMPLE = "https://huggingface.co/MrDawg/ToothBrushing/resolve/main/ToothBrushing.zip?download=true"
INFO_EXAMPLES = f"""
<span style="font-size: 0.70rem; color: var(--body-text-color-subdued, #6b7280); display: block; margin: 3px 0 6px 2px; word-break: break-all; line-height: 1.4;">
  💡 Provide a <code style="font-size: 0.60rem; padding: 1px 3px; background: var(--background-fill-secondary, #f3f4f6); border-radius: 3px;">.zip</code> e.g. <code>{ZIP_EXAMPLE}</code> or separate <code style="font-size: 0.60rem; padding: 1px 3px; background: var(--background-fill-secondary, #f3f4f6); border-radius: 3px;">.pth, .index</code> e.g. <code>{TEST_MODEL}</code>
</span>
"""
RESOURCES = "- You can also try `RVC⚡ZERO` in Colab’s free tier, which provides free GPU [link](https://github.com/R3gm/rvc_zero_ui?tab=readme-ov-file#rvczero)."
DELETE_CACHE_TIME = (3200, 3200) if IS_ZERO_GPU else (86400, 86400)

PITCH_ALGO_OPT = [
    "pm",
    "harvest",
    "crepe",
    "rmvpe",
    "rmvpe+",
]

hf_api = HfApi()

def get_hf_likes(repo_id: str):
    """Fetches and caches the repository like count using the official Hugging Face Hub API."""
    if not repo_id or repo_id == "Community / Hugging Face":
        return "N/A"

    if repo_id in LIKES_CACHE:
        return LIKES_CACHE[repo_id]

    try:
        info = hf_api.model_info(repo_id=repo_id, timeout=3)
        likes_count = info.likes if info.likes is not None else 0
        LIKES_CACHE[repo_id] = f"{likes_count:,}"
        return LIKES_CACHE[repo_id]
    except Exception:
        return "N/A"


def update_search_selection(selected_url):
    if not selected_url:
        return "", (
            "<div style='padding: 12px 16px; border-radius: 8px; border: 1px dashed var(--border-color-primary, #d1d5db); "
            "background: var(--background-fill-secondary, #f9fafb); color: var(--body-text-color-subdued, #6b7280); font-size: 0.9rem; text-align: center;'>"
            "🔍 <em>No model selected yet. Search above and pick a model from the list.</em>"
            "</div>"
        )

    model_name = "Unknown Model"
    repo_name = "Community / Hugging Face"
    repo_url = "https://huggingface.co"
    direct_url = selected_url
    repo_id = None

    if voice_finder.conn:
        try:
            cursor = voice_finder.conn.cursor()
            cursor.execute(
                "SELECT FILENAME, MODEL_ID, PARSED_URL FROM files WHERE PARSED_URL = ? LIMIT 1",
                (selected_url,)
            )
            row = cursor.fetchone()
            if row:
                model_name = row[0]
                repo_id = row[1]
                direct_url = row[2]
                repo_name = repo_id
                repo_url = f"https://huggingface.co/{repo_id}"
        except Exception as e:
            print(f"Error resolving model details from DB: {e}")

    likes = get_hf_likes(repo_id) if repo_id else "N/A"

    info_card = f"""
<div style="padding: 14px 16px; border-radius: 10px; border: 1px solid var(--border-color-primary, #e5e7eb); background: var(--background-fill-secondary, #f8fafc); margin-top: 6px;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
    <div style="font-weight: 700; font-size: 1rem; color: var(--body-text-color, #0f172a);">
      📦 {model_name}
    </div>
    <div style="background: #fef3c7; color: #b45309; border: 1px solid #fde68a; font-size: 0.8rem; font-weight: 600; padding: 2px 10px; border-radius: 9999px; display: inline-flex; align-items: center; gap: 4px;">
      ⭐ {likes} Likes
    </div>
  </div>
  <div style="font-size: 0.86rem; line-height: 1.6; color: var(--body-text-color-subdued, #334155);">
    <div><strong>🏛️ Repository:</strong> <a href="{repo_url}" target="_blank" style="color: #2563eb; text-decoration: underline; font-weight: 500;">{repo_name} ↗</a></div>
    <div style="margin-top: 4px; word-break: break-all;"><strong>🔗 Direct URL:</strong> <code style="font-size: 0.8rem; background: var(--background-fill-primary, #ffffff); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border-color-primary, #e2e8f0);">{direct_url}</code></div>
  </div>
</div>
"""
    return direct_url, info_card


class ModelVoiceFinder:
    def __init__(
        self,
        database_url="https://raw.githubusercontent.com/R3gm/database_zip_files/main/archive/database.csv",
        database_path="database.csv",
    ):
        self.database_url = database_url
        self.database_path = database_path
        self.conn = None
        self._load_and_init()

    def _clean_file_url(self, val):
        if pd.isna(val):
            return ""
        if isinstance(val, list):
            return ", ".join(map(str, val))
        if isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return ", ".join(map(str, parsed))
            except Exception:
                return val
        return str(val)

    def _normalize(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        return re.sub(r"[+()\-_/.]", " ", text)

    def _load_and_init(self):
        try:
            if not os.path.exists(self.database_path):
                print("Downloading community voices database...")
                req = urllib.request.Request(self.database_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(self.database_path, "wb") as f:
                    shutil.copyfileobj(resp, f)

            df = pd.read_csv(self.database_path)
            df["FILENAME_NORM"] = df["FILENAME"].apply(self._normalize)
            df["PARSED_URL"] = df["PARSED_URL"].apply(self._clean_file_url)
            df = df.reset_index(drop=True)
            df["orig_index"] = df.index

            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
            df.to_sql("files", self.conn, index=False, if_exists="replace")

            # Create indices for instant search speed
            cursor = self.conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename_norm ON files(FILENAME_NORM)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_parsed_url ON files(PARSED_URL)")
            self.conn.commit()

            print("Community voices database preloaded and indexed successfully.")
        except Exception as e:
            print(f"Error initializing ModelVoiceFinder: {e}")

    def search(self, query: str):
        if not self.conn or not query or not query.strip():
            return gr.update(choices=[], value=None), "", update_search_selection(None)[1]

        keywords = self._normalize(query).split()
        if not keywords:
            return gr.update(choices=[], value=None), "", update_search_selection(None)[1]

        # Sanitize tokens to prevent SQL breaking characters
        safe_keywords = [re.sub(r"[^\w\s]", "", k) for k in keywords if k.strip()]
        if not safe_keywords:
            return gr.update(choices=[], value=None), "", update_search_selection(None)[1]

        whole_conditions = " AND ".join([
            f"(FILENAME_NORM LIKE '% {k} %' OR FILENAME_NORM LIKE '{k} %' OR FILENAME_NORM LIKE '% {k}' OR FILENAME_NORM = '{k}')"
            for k in safe_keywords
        ])
        partial_conditions = " AND ".join([f"FILENAME_NORM LIKE '%{k}%'" for k in safe_keywords])

        sql = f"""
        SELECT FILENAME, PARSED_URL, MODEL_ID,
               CASE WHEN {whole_conditions} THEN 1 ELSE 0 END AS whole_match
        FROM files
        WHERE {partial_conditions}
        ORDER BY whole_match DESC, orig_index ASC
        LIMIT 150;
        """

        try:
            df_res = pd.read_sql(sql, self.conn)
            if df_res.empty:
                gr.Info("I searched everywhere, but found nothing matching that...")
                empty_card = (
                    "<div style='padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border-color-primary, #e5e7eb); "
                    "background: var(--background-fill-secondary, #f9fafb); color: #ef4444; font-size: 0.9rem; text-align: center;'>"
                    "❌ <em>No voice models found matching your search.</em>"
                    "</div>"
                )
                return gr.update(choices=[], value=None), "", empty_card

            choices = []
            for row in df_res.itertuples(index=False):
                display_name = f"{row.FILENAME} (Repo: {row.MODEL_ID})"
                choices.append((display_name, row.PARSED_URL))

            first_match_url = choices[0][1] if choices else None
            direct_url, info_card = update_search_selection(first_match_url)
            return gr.update(choices=choices, value=first_match_url), direct_url, info_card
        except Exception as e:
            print(f"Search error: {e}")
            return gr.update(choices=[], value=None), "", "*Search error occurred.*"


voice_finder = ModelVoiceFinder()


def check_model_safety(file_path: str):
    """Statically checks model or archive integrity. Raises ValueError if unsafe."""
    if not file_path or not os.path.exists(file_path):
        print(f"Skip file: {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".pth", ".pt", ".bin", ".zip", ".pkl"):
        try:
            result = scan_file_path(file_path)
            if result and result.infected_files > 0:
                raise ValueError(
                    f"Integrity check failed: '{os.path.basename(file_path)}' contains unsupported or unsafe structures."
                )
        except ValueError:
            raise
        except Exception as e:
            print(f"Integrity check skipped for {file_path}: {e}")


def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_url_hash(url_data):
    if not url_data:
        return None
    if "," in url_data:
        a_, b_ = url_data.split(",")
        a_, b_ = a_.strip().replace("/blob/", "/resolve/"), b_.strip().replace("/blob/", "/resolve/")
        url_string = f"{a_}_{b_}"
    else:
        url_string = url_data.strip().replace("/blob/", "/resolve/")
    return hashlib.sha256(url_string.encode()).hexdigest()


def cache_gradio_paths(processed_url, model_file, index_file):
    if not processed_url or not model_file:
        return
        
    url_hash = get_url_hash(processed_url)
    
    def extract_path(f):
        if not f: return None
        if isinstance(f, str): return f
        if isinstance(f, dict): return f.get("name", "")
        return getattr(f, "name", str(f))

    m_path = extract_path(model_file)
    i_path = extract_path(index_file)
    
    if m_path:
        URL_CACHE[url_hash] = (m_path, i_path)


async def get_voices_list(proxy=None):
    """Print all available voices."""
    from edge_tts import list_voices
    voices = await list_voices(proxy=proxy)
    voices = sorted(voices, key=lambda voice: voice["ShortName"])

    table = [
        {
            "ShortName": voice["ShortName"],
            "Gender": voice["Gender"],
            "ContentCategories": ", ".join(voice["VoiceTag"]["ContentCategories"]),
            "VoicePersonalities": ", ".join(voice["VoiceTag"]["VoicePersonalities"]),
            "FriendlyName": voice["FriendlyName"],
        }
        for voice in voices
    ]

    return table


def find_files(directory):
    file_paths = []
    for filename in os.listdir(directory):
        # Check if the file has the desired extension
        if filename.endswith('.pth') or filename.endswith('.zip') or filename.endswith('.index'):
            # If yes, add the file path to the list
            file_paths.append(os.path.join(directory, filename))

    return file_paths


def unzip_in_folder(my_zip, my_dir):
    with zipfile.ZipFile(my_zip) as zip:
        for zip_info in zip.infolist():
            if zip_info.is_dir():
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, my_dir)


def find_my_model(a_, b_):
    if a_ is None or a_.endswith(".pth"):
        if a_ and a_.endswith(".pth"):
            check_model_safety(a_)
        return a_, b_

    input_hash = None
    if a_ and os.path.exists(a_):
        input_hash = get_file_hash(a_)
        if b_ and os.path.exists(b_):
            input_hash += "_" + get_file_hash(b_)

    if input_hash and input_hash in MODEL_CACHE:
        cached_model, cached_index = MODEL_CACHE[input_hash]
        model_exists = cached_model and os.path.exists(cached_model)
        index_exists = (cached_index is None) or os.path.exists(cached_index)

        if model_exists and index_exists:
            check_model_safety(cached_model)
            gr.Info(f"Model: {os.path.basename(cached_model)}")
            if cached_index:
                gr.Info(f"Index: {os.path.basename(cached_index)}")
            return cached_model, cached_index
        else:
            del MODEL_CACHE[input_hash]

    txt_files = []
    for base_file in [a_, b_]:
        if base_file is not None and base_file.endswith(".txt"):
            txt_files.append(base_file)

    directory = os.path.dirname(a_)

    for txt in txt_files:
        with open(txt, 'r') as file:
            first_line = file.readline()

        url_to_download = first_line.strip()
        ensure_valid_file(url_to_download)

        hf_download_file(url_to_download, directory)

    for f in find_files(directory):
        if f.endswith(".zip"):
            check_model_safety(f)
            unzip_in_folder(f, directory)

    model = None
    index = None
    end_files = find_files(directory)

    for ff in end_files:
        if ff.endswith(".pth"):
            check_model_safety(ff)
            model = ff
            gr.Info(f"Model: {os.path.basename(ff)}")
        if ff.endswith(".index"):
            index = ff
            gr.Info(f"Index: {os.path.basename(ff)}")

    if not model:
        gr.Error("No valid .pth model file found,")

    if not index:
        gr.Warning("Couldn't find an index file... We'll just have to proceed without it.")

    if model and input_hash:
        MODEL_CACHE[input_hash] = (model, index)

    return model, index


def validate_url(url: str) -> str:
    """Validate URL protocol and host."""
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty.")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Invalid protocol '{parsed.scheme}'. Only HTTPS is allowed.")

    hostname = (parsed.hostname or "").lower()
    if not any(hostname == d or hostname.endswith("." + d) for d in ALLOWED_DOMAINS):
        raise ValueError("Only Hugging Face links are allowed here.")

    return url


def get_supported_audio_video_extensions():
    try:
        subtitle_codecs = set()
        decoders_res = subprocess.run(
            ["ffmpeg", "-decoders"], capture_output=True, text=True, check=True
        )
        for line in decoders_res.stdout.splitlines():
            line_str = line.strip()
            if len(line_str) > 7 and line_str[0] in ("V", "A", "S"):
                parts = line_str.split()
                if len(parts) >= 2:
                    media_type = line_str[0]
                    codec_name = parts[1].lower()
                    if media_type == "S":
                        subtitle_codecs.add(codec_name)

        demuxers_res = subprocess.run(
            ["ffmpeg", "-demuxers"], capture_output=True, text=True, check=True
        )
        extensions = set()
        for line in demuxers_res.stdout.splitlines():
            line_str = line.strip()
            if line_str.startswith("D"):
                parts = line_str.split(maxsplit=2)
                if len(parts) >= 2:
                    demux_names = parts[1].split(",")
                    description = parts[2].lower() if len(parts) >= 3 else ""

                    if any(
                        sub_kw in description
                        for sub_kw in ["subtitle", "caption", "teletext", "lyrics"]
                    ):
                        continue

                    for ext in demux_names:
                        clean_ext = ext.strip().lower()
                        if clean_ext in subtitle_codecs:
                            continue
                        if clean_ext and clean_ext.isalnum():
                            extensions.add(f".{clean_ext}")

        return sorted(list(extensions))
    except Exception as e:
        print(f"Error querying ffmpeg: {e}")
        return [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".alac", ".caf", ".amr"]


def ensure_valid_file(url):
    url = validate_url(url)

    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            content_length = response.headers.get("Content-Length")

        if content_length is None:
            raise ValueError("Unable to read file info from url. The link might be invalid or unreachable.")

        file_size = int(content_length)
        # print("debug", url, file_size)
        if file_size > 900000000 and IS_ZERO_GPU:
            raise ValueError("That file is way too big. 900 MB is the absolute limit.")

        return file_size

    except urllib.error.HTTPError as err:
        raise ValueError(
            f"HTTP Error {err.code} ({err.reason}): The file at the provided URL does not exist, is private, or has been deleted from Hugging Face. Please verify the link."
        )

    except Exception as e:
        raise e


def clear_files(directory):
    time.sleep(15)
    # print(f"Clearing files: {directory}.")
    shutil.rmtree(directory)


def get_my_model(url_data, progress=gr.Progress(track_tqdm=True)):
    if not url_data:
        return None, None, None

    url_hash = get_url_hash(url_data)
    if url_hash in URL_CACHE:
        cached_model, cached_index = URL_CACHE[url_hash]
        if cached_model and os.path.exists(cached_model) and ((cached_index is None) or os.path.exists(cached_index)):
            gr.Info(f"Model: {os.path.basename(cached_model)}")
            if cached_index is not None:
                gr.Info(f"Index: {os.path.basename(cached_index)}")
            return cached_model, cached_index, url_data
        else:
            del URL_CACHE[url_hash]

    if "," in url_data:
        a_, b_ = url_data.split(",")
        a_, b_ = a_.strip().replace("/blob/", "/resolve/"), b_.strip().replace("/blob/", "/resolve/")
    else:
        a_, b_ = url_data.strip().replace("/blob/", "/resolve/"), None

    out_dir = "downloads"
    folder_download = str(random.randint(1000, 9999))
    directory = os.path.join(out_dir, folder_download)
    os.makedirs(directory, exist_ok=True)

    try:
        valid_url = [a_] if not b_ else [a_, b_]
        for link in valid_url:
            ensure_valid_file(link)
            hf_download_file(link, directory)

        for f in find_files(directory):
            if f.endswith(".zip"):
                check_model_safety(f)
                unzip_in_folder(f, directory)

        model = None
        index = None
        end_files = find_files(directory)

        for ff in end_files:
            if ff.endswith(".pth"):
                check_model_safety(ff)
                model = ff
                gr.Info(f"Model: {os.path.basename(ff)}")
            if ff.endswith(".index"):
                index = ff
                gr.Info(f"Index: {os.path.basename(ff)}")

        if not model:
            raise ValueError("No valid .pth model file found.")

        if not index:
            gr.Info("Couldn't find an index file... We'll just have to proceed without it.")
        else:
            index = os.path.abspath(index)

        return os.path.abspath(model), index, url_data

    except Exception as e:
        raise e
    finally:
        # time.sleep(10)
        # shutil.rmtree(directory)
        t = threading.Thread(target=clear_files, args=(directory,))
        t.start()


def add_audio_effects(audio_list, type_output):
    print("Audio effects")

    result = []
    for audio_path in audio_list:
        try:
            output_path = f'{os.path.splitext(audio_path)[0]}_effects.{type_output}'

            # Initialize audio effects plugins
            board = Pedalboard(
                [
                    HighpassFilter(),
                    Compressor(ratio=4, threshold_db=-15),
                    Reverb(room_size=0.10, dry_level=0.8, wet_level=0.2, damping=0.7)
                ]
            )

            # Temporary WAV to hold processed data before exporting
            temp_wav = f'{os.path.splitext(audio_path)[0]}_temp.wav'

            with AudioFile(audio_path) as f:
                with AudioFile(temp_wav, 'w', f.samplerate, f.num_channels) as o:
                    while f.tell() < f.frames:
                        chunk = f.read(int(f.samplerate))
                        effected = board(chunk, f.samplerate, reset=False)
                        o.write(effected)

            # Convert with pydub to desired output type
            audio_seg = AudioSegment.from_file(temp_wav, format=type_output)
            audio_seg.export(output_path, format=type_output, bitrate=("320k" if type_output == "mp3" else None))

            # Clean up temp file
            os.remove(temp_wav)

            result.append(output_path)
        except Exception as e:
            traceback.print_exc()
            print(f"Error noisereduce: {str(e)}")
            result.append(audio_path)

    return result


def apply_noisereduce(audio_list, type_output):
    # https://github.com/sa-if/Audio-Denoiser
    print("Noice reduce")

    result = []
    for audio_path in audio_list:
        out_path = f"{os.path.splitext(audio_path)[0]}_noisereduce.{type_output}"

        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)

            # Convert audio to numpy array
            samples = np.array(audio.get_array_of_samples())

            # Reduce noise
            reduced_noise = nr.reduce_noise(samples, sr=audio.frame_rate, prop_decrease=0.6)

            # Convert reduced noise signal back to audio
            reduced_audio = AudioSegment(
                reduced_noise.tobytes(), 
                frame_rate=audio.frame_rate, 
                sample_width=audio.sample_width,
                channels=audio.channels
            )

            # Save reduced audio to file
            reduced_audio.export(out_path, format=type_output, bitrate=("320k" if type_output == "mp3" else None))
            result.append(out_path)

        except Exception as e:
            traceback.print_exc()
            print(f"Error noisereduce: {str(e)}")
            result.append(audio_path)

    return result


@spaces.GPU()
def convert_now(audio_files, type_output, steps, params_tag):
    global converter

    try:
        
        random_tag = params_tag["tag"]
        converter.apply_conf(**params_tag)
        
        for step in range(steps):
            if step > 0:
                conf = converter.model_config[random_tag]
                conf["pitch_lvl"] = 0                              # Pitch is already at target
                # conf["respiration_median_filtering"] = 0           # Avoid flattening natural vibrato
                # conf["envelope_ratio"] = 1.0                       # Skip redundant RMS volume matching
                conf["index_influence"] = round((conf["index_influence"] / 5), 2)  # Reduce artifact stacking
                # conf["consonant_breath_protection"] = 0.5          # Avoid cutting consonants twice
    
            audio_files = converter(
                audio_files,
                random_tag,
                overwrite=False,
                parallel_workers=(2 if IS_COLAB else 8),
                type_output=type_output,
                show_progress=False,
            )

        return audio_files, None
    except Exception as e:
        traceback.print_exc()
        return [], str(e)    


def run(
    audio_files,
    file_m,
    pitch_alg,
    pitch_lvl,
    file_index,
    index_inf,
    r_m_f,
    e_r,
    c_b_p,
    active_noise_reduce,
    audio_effects,
    type_output,
    steps,
):
    if not audio_files:
        raise gr.Error("You didn't provide any audio... How am I supposed to convert nothing?")

    if isinstance(audio_files, str):
        audio_files = [audio_files]

    # try:
    #     duration_base = librosa.get_duration(filename=audio_files[0])
    #     print("Duration:", duration_base)
    # except Exception as e:
    #     print(e)

    if file_m is not None and (file_m.endswith(".txt") or file_m.endswith(".zip")):
        file_m, file_index = find_my_model(file_m, file_index)

    # 1. Verify files exist using template
    files_to_check = [("Audio", f) for f in audio_files] + [("Model", file_m)]
    if file_index:
        files_to_check.append(("Index", file_index))

    for label, path in files_to_check:
        if not path or not os.path.exists(path):
            raise gr.Error(f"The Space may have restarted or your {label.lower()} file was not completely uploaded... Just re-upload your {label.lower()}.")

    random_tag = "USER_" + str(random.randint(10000000, 99999999))

    params_tag = dict(
        tag=random_tag,
        file_model=file_m,
        pitch_algo=pitch_alg,
        pitch_lvl=pitch_lvl,
        file_index=file_index,
        index_influence=index_inf,
        respiration_median_filtering=r_m_f,
        envelope_ratio=e_r,
        consonant_breath_protection=c_b_p,
        resample_sr=0,
    )
    # time.sleep(0.1)

    result, error_msg = convert_now(audio_files, type_output, steps, params_tag)

    if error_msg is not None:
        raise gr.Error(f"Error: {error_msg}")        
    if not result:
        raise gr.Error("Conversion failed due to an unexpected error. Please check your files and settings, then try again.")

    if active_noise_reduce:
        result = apply_noisereduce(result, type_output)

    if audio_effects:
        result = add_audio_effects(result, type_output)

    return result


def clear_player():
    return None


def load_first_audio(output_files, play_audio):
    if not play_audio or not output_files:
        return None
    first_file = output_files[0]
    if isinstance(first_file, dict):
        first_file = first_file.get("name")
    return first_file


def mic_conf():
    return gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Record Audio",
    )


def audio_conf():
    return gr.File(
        label="Target Audio File(s)",
        file_count="multiple",
        type="filepath",
        container=True,
        file_types=supported_extensions,
    )


def model_conf():
    return gr.File(
        label="Model File",
        type="filepath",
        height=130,
        file_types=[".txt", ".pth", ".zip"],
    )


def index_conf():
    return gr.File(
        label="Index File",
        type="filepath",
        height=130,
        file_types=[".index", ".txt"],
    )


def pitch_algo_conf():
    return gr.Dropdown(
        PITCH_ALGO_OPT,
        value=PITCH_ALGO_OPT[4],
        label="Pitch algorithm",
        interactive=True,
    )


def pitch_lvl_conf():
    return gr.Slider(
        label="Pitch level",
        minimum=-24,
        maximum=24,
        step=1,
        value=0,
        interactive=True,
    )


def index_inf_conf():
    return gr.Slider(
        minimum=0,
        maximum=1,
        label="Index influence",
        value=0.75,
        interactive=True,
    )


def respiration_filter_conf():
    return gr.Slider(
        minimum=0,
        maximum=7,
        label="Respiration median filtering",
        value=3,
        step=1,
        interactive=True,
    )


def envelope_ratio_conf():
    return gr.Slider(
        minimum=0,
        maximum=1,
        label="Envelope ratio",
        value=0.25,
        interactive=True,
    )


def consonant_protec_conf():
    return gr.Slider(
        minimum=0,
        maximum=0.5,
        label="Consonant breath protection",
        value=0.5,
        interactive=True,
    )


def steps_conf():
    return gr.Slider(
        minimum=1,
        maximum=3,
        label="Sub-conversion Cycles",
        value=1,
        step=1,
        interactive=True,
    )


def format_output_gui():
    return gr.Dropdown(
        label="Format output:",
        choices=["wav", "mp3", "flac"],
        value="wav",
        interactive=True,
    )


def denoise_conf():
    return gr.Checkbox(
        False,
        label="Denoise",
        container=False,
    )


def effects_conf():
    return gr.Checkbox(
        False,
        label="Reverb",
        container=False,
    )


def player_conf():
    return gr.Checkbox(
        False,
        label="Audio Player",
        container=False,
    )


def button_conf():
    return gr.Button(
        "⚡ Inference",
        variant="primary",
        elem_classes=["generate-btn"],
    )


def output_conf():
    return gr.File(
        label="Result Audio",
        file_count="multiple",
        interactive=False,
    )


def tts_voice_conf():
    return gr.Dropdown(
        label="TTS Voice",
        choices=voices,
        value="en-US-EmmaMultilingualNeural-Female",
        interactive=True,
    )


def tts_text_conf():
    return gr.Textbox(
        value="",
        placeholder="Write the text here...",
        label="Text",
        lines=3,
    )


def tts_button_conf():
    return gr.Button(
        "Process TTS",
        variant="secondary",
    )


def sound_gui():
    return gr.Audio(
        value=None,
        type="filepath",
        autoplay=True,
        visible=False,
        interactive=False,
        label="Audio Preview",
    )


def down_url_conf():
    return gr.Textbox(
        value="",
        placeholder="Paste a Hugging Face link here...",
        label="Model URL",
        lines=1,
    )


def down_button_conf():
    return gr.Button(
        "⬇️ Download & Load Model",
        variant="secondary",
    )


def infer_tts_audio(tts_voice, tts_text, play_audio):
    out_dir = "output"
    folder_tts = "USER_" + str(random.randint(10000, 99999))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, folder_tts), exist_ok=True)
    out_path = os.path.join(out_dir, folder_tts, "tts.mp3")

    asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(out_path))
    player_audio = out_path if play_audio else None
    return [out_path], player_audio


def sync_mic_audio(mic_path):
    if not mic_path:
        return gr.update()
    return [mic_path]


def export_rvc_settings_json(
    audio_tab,
    model_tab,
    tts_voice,
    down_url,
    algo,
    algo_lvl,
    indx_inf,
    res_fc,
    envel_r,
    const,
    steps,
    format_out,
    denoise,
    effects,
    player,
):
    settings_dict = {
        "audio_tab": str(audio_tab) if audio_tab else "tab_audio_upload",
        "model_tab": str(model_tab) if model_tab else "tab_model_upload",
        "tts_voice": tts_voice,
        "down_url": down_url,
        "pitch_algo": algo,
        "pitch_lvl": int(algo_lvl),
        "index_influence": float(indx_inf),
        "respiration_filter": int(res_fc),
        "envelope_ratio": float(envel_r),
        "consonant_protection": float(const),
        "steps": int(steps),
        "format_output": format_out,
        "denoise": bool(denoise),
        "effects": bool(effects),
        "player": bool(player),
    }
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"rvc_settings_{uuid.uuid4().hex[:8]}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4, ensure_ascii=False)
    
    return file_path


def parse_rvc_settings_file(file_obj):
    if not file_obj:
        return [gr.update() for _ in range(17)]

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        gr.Warning("I couldn't read that settings file... Make sure it's a valid JSON.")
        return [gr.update() for _ in range(17)]

    if not isinstance(data, dict):
        gr.Warning("This settings file format is all wrong...")
        return [gr.update() for _ in range(17)]

    saved_algo = data.get("pitch_algo")
    algo_val = saved_algo if saved_algo in PITCH_ALGO_OPT else gr.update()
    saved_aud_tab = data.get("audio_tab", "tab_audio_upload")
    saved_mod_tab = data.get("model_tab", "tab_model_upload")

    gr.Info("Fine, your settings have been loaded.")
    return (
        gr.Tabs(selected=saved_aud_tab),
        gr.Tabs(selected=saved_mod_tab),
        saved_aud_tab,
        saved_mod_tab,
        data.get("tts_voice", gr.update()),
        data.get("down_url", ""),
        algo_val,
        data.get("pitch_lvl", gr.update()),
        data.get("index_influence", gr.update()),
        data.get("respiration_filter", gr.update()),
        data.get("envelope_ratio", gr.update()),
        data.get("consonant_protection", gr.update()),
        data.get("steps", gr.update()),
        data.get("format_output", gr.update()),
        data.get("denoise", gr.update()),
        data.get("effects", gr.update()),
        data.get("player", gr.update()),
    )


# CSS rules to hide element visually while keeping DOM node accessible
BASE_CSS = """
#download_settings_json_hidden {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0px !important;
    height: 0px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    overflow: hidden !important;
}
"""

R3_THEME_CSS = """
button.generate-btn {
    width: 100% !important;
    background-color: #111111 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease !important;
    display: inline-flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 0.5rem !important;
    box-shadow: none !important;
}
button.generate-btn:hover {
    background-color: #222222 !important;
}
.dark button.generate-btn {
    background-color: #ffffff !important;
    color: #111111 !important;
}
.dark button.generate-btn:hover {
    background-color: #e5e5e5 !important;
}
@media (prefers-color-scheme: dark) {
    body:not(.light) button.generate-btn {
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    body:not(.light) button.generate-btn:hover {
        background-color: #e5e5e5 !important;
    }
}
.compact-btn-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 0.5rem !important;
    flex-wrap: wrap !important;
    margin-top: 0.5rem !important;
}
.compact-btn,
.compact-btn button,
.gradio-container button.secondary {
    width: auto !important;
    min-width: unset !important;
    max-width: fit-content !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 6px 12px !important;
    min-height: 32px !important;
    height: auto !important;
    line-height: 1.2 !important;
    text-align: center !important;
    white-space: normal !important;
    font-size: 0.813rem !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
    background-color: #111111 !important;
    border: none !important;
    color: #ffffff !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}
.compact-btn:hover,
.compact-btn button:hover,
.gradio-container button.secondary:hover {
    background-color: #222222 !important;
}
.dark .compact-btn,
.dark .compact-btn button,
.dark .gradio-container button.secondary {
    background-color: #ffffff !important;
    border: none !important;
    color: #111111 !important;
}
.dark .compact-btn:hover,
.dark .compact-btn button:hover,
.dark .gradio-container button.secondary:hover {
    background-color: #e5e5e5 !important;
}
@media (prefers-color-scheme: dark) {
    body:not(.light) .compact-btn,
    body:not(.light) .compact-btn button,
    body:not(.light) .gradio-container button.secondary {
        background-color: #ffffff !important;
        border: none !important;
        color: #111111 !important;
    }
    body:not(.light) .compact-btn:hover,
    body:not(.light) .compact-btn button:hover,
    body:not(.light) .gradio-container button.secondary:hover {
        background-color: #e5e5e5 !important;
    }
}
:root {
    --component-border-color: var(--neutral-200, rgba(128, 128, 128, 0.18));
    --component-bg-color: var(--neutral-50, rgba(128, 128, 128, 0.03));
}
.dark {
    --component-border-color: var(--neutral-800, rgba(255, 255, 255, 0.12));
    --component-bg-color: var(--neutral-900, rgba(255, 255, 255, 0.03));
}
.gradio-container .block,
.gradio-container .panel,
.gradio-container .form,
.gradio-container fieldset {
    border: 1px solid var(--component-border-color) !important;
    background-color: var(--component-bg-color) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
.gradio-container .markdown,
.gradio-container .prose,
.gradio-container .block.prose,
.gradio-container div[class*="markdown"] {
    border: none !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}
"""

APP_CSS = (BASE_CSS + R3_THEME_CSS) if IS_R3_THEME else BASE_CSS

supported_extensions = get_supported_audio_video_extensions()
print("Supported extensions found:", supported_extensions)


def get_gui():
    with gr.Blocks(fill_width=True, fill_height=False, delete_cache=DELETE_CACHE_TIME) as app:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        processed_url_state = gr.State()
        active_audio_tab = gr.State("tab_audio_upload")
        active_model_tab = gr.State("tab_model_upload")

        with gr.Row():
            # Left Column: Inputs & Models
            with gr.Column(scale=1):
                with gr.Tabs(selected="tab_audio_upload") as audio_tabs:
                    with gr.Tab("📁 Upload", id="tab_audio_upload") as tab_aud_up:
                        gr.Markdown("Upload your audio files directly to the box below.")
                    with gr.Tab("🗣️ TTS", id="tab_audio_tts") as tab_aud_tts:
                        tts_text = tts_text_conf()
                        tts_voice = tts_voice_conf()
                        tts_button = tts_button_conf()
                    with gr.Tab("🎙️ Record", id="tab_audio_rec") as tab_aud_rec:
                        mic_aud = mic_conf()

                aud = audio_conf()

                with gr.Tabs(selected="tab_model_upload") as model_tabs:
                    with gr.Tab("📁 Upload Model", id="tab_model_upload") as tab_mod_up:
                        gr.Markdown("Upload the model and optional index files below.")
                    with gr.Tab("🌐 Direct URL", id="tab_model_url") as tab_mod_url:
                        gr.Markdown(INFO_EXAMPLES)
                        down_url_gui = down_url_conf()
                        down_button_gui = down_button_conf()
                    with gr.Tab("🔍 Search Community", id="tab_model_search") as tab_mod_search:
                        with gr.Row():
                            search_query = gr.Textbox(label="Search Query", placeholder="Hatsune Miku", scale=3, lines=1)
                            search_btn = gr.Button("🔍 Search", variant="secondary", scale=1)
                        search_results = gr.Dropdown(label="Search Results", choices=[], interactive=True)
                        model_info_md = gr.Markdown(value="*Select a model from the dropdown to view details.*")
                        search_down_btn = gr.Button("⬇️ Download & Load Model", variant="secondary")

                with gr.Row():
                    model = model_conf()
                    indx = index_conf()

            # Right Column: Settings, Actions & Outputs
            with gr.Column(scale=1):
                with gr.Accordion(label="Advanced settings", open=False):
                    with gr.Row(elem_classes=["compact-btn-row"]):
                        load_settings_btn = gr.UploadButton(
                            "📂 Load Settings (JSON)",
                            file_types=[".json"],
                            file_count="single",
                            size="sm",
                            elem_classes=["compact-btn"],
                        )
                        download_json_btn = gr.Button(
                            "💾 Download Settings JSON",
                            size="sm",
                            elem_classes=["compact-btn"],
                        )
                        # Keep DOM node alive and hide with CSS
                        download_json_hidden = gr.DownloadButton(
                            visible=True,
                            elem_id="download_settings_json_hidden",
                        )

                    with gr.Row():
                        algo = pitch_algo_conf()
                        algo_lvl = pitch_lvl_conf()

                    with gr.Row():
                        indx_inf = index_inf_conf()
                        steps_gui = steps_conf()

                    with gr.Row():
                        res_fc = respiration_filter_conf()
                        envel_r = envelope_ratio_conf()

                    with gr.Row():
                        const = consonant_protec_conf()
                        format_out = format_output_gui()

                    with gr.Row():
                        denoise_gui = denoise_conf()
                        effects_gui = effects_conf()

                button_base = button_conf()

                with gr.Row():
                    gr.Markdown("### 🎧 Result Audio")
                    player_gui = player_conf()

                player_audio = sound_gui()
                output_base = output_conf()

        # Tab Selection Tracking
        tab_aud_up.select(lambda: "tab_audio_upload", outputs=[active_audio_tab], api_visibility="private")
        tab_aud_tts.select(lambda: "tab_audio_tts", outputs=[active_audio_tab], api_visibility="private")
        tab_aud_rec.select(lambda: "tab_audio_rec", outputs=[active_audio_tab], api_visibility="private")

        tab_mod_up.select(lambda: "tab_model_upload", outputs=[active_model_tab], api_visibility="private")
        tab_mod_url.select(lambda: "tab_model_url", outputs=[active_model_tab], api_visibility="private")
        tab_mod_search.select(lambda: "tab_model_search", outputs=[active_model_tab], api_visibility="private")

        # Events Wiring
        mic_aud.change(
            fn=sync_mic_audio,
            inputs=[mic_aud],
            outputs=[aud],
            api_visibility="private",
        )

        tts_button.click(
            fn=infer_tts_audio,
            inputs=[tts_voice, tts_text, player_gui],
            outputs=[aud, player_audio],
            api_visibility="private",
        )

        player_gui.change(
            fn=lambda val: gr.update(visible=val),
            inputs=[player_gui],
            outputs=[player_audio],
            api_visibility="private",
        )

        search_btn.click(
            fn=voice_finder.search,
            inputs=[search_query],
            outputs=[search_results, down_url_gui, model_info_md],
            api_visibility="private",
        )
        search_query.submit(
            fn=voice_finder.search,
            inputs=[search_query],
            outputs=[search_results, down_url_gui, model_info_md],
            api_visibility="private",
        )

        search_results.select(
            fn=update_search_selection,
            inputs=[search_results],
            outputs=[down_url_gui, model_info_md],
            api_visibility="private",
        )

        down_button_gui.click(
            fn=get_my_model,
            inputs=[down_url_gui],
            outputs=[model, indx, processed_url_state],
            api_visibility="private",
        ).success(
            fn=cache_gradio_paths,
            inputs=[processed_url_state, model, indx],
            outputs=[],
            api_visibility="private",
        )

        search_down_btn.click(
            fn=get_my_model,
            inputs=[down_url_gui],
            outputs=[model, indx, processed_url_state],
            api_visibility="private",
        ).success(
            fn=cache_gradio_paths,
            inputs=[processed_url_state, model, indx],
            outputs=[],
            api_visibility="private",
        )

        download_json_btn.click(
            fn=export_rvc_settings_json,
            inputs=[
                active_audio_tab,
                active_model_tab,
                tts_voice,
                down_url_gui,
                algo,
                algo_lvl,
                indx_inf,
                res_fc,
                envel_r,
                const,
                steps_gui,
                format_out,
                denoise_gui,
                effects_gui,
                player_gui,
            ],
            outputs=[download_json_hidden],
            api_visibility="private",
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { const el = document.querySelector('#download_settings_json_hidden button') || document.querySelector('#download_settings_json_hidden a') || document.querySelector('#download_settings_json_hidden'); if (el) el.click(); }",
        )

        load_settings_btn.upload(
            fn=parse_rvc_settings_file,
            inputs=[load_settings_btn],
            outputs=[
                audio_tabs,
                model_tabs,
                active_audio_tab,
                active_model_tab,
                tts_voice,
                down_url_gui,
                algo,
                algo_lvl,
                indx_inf,
                res_fc,
                envel_r,
                const,
                steps_gui,
                format_out,
                denoise_gui,
                effects_gui,
                player_gui,
            ],
            api_visibility="private",
        ).then(
            fn=get_my_model,
            inputs=[down_url_gui],
            outputs=[model, indx, processed_url_state],
            api_visibility="private",
        ).success(
            fn=cache_gradio_paths,
            inputs=[processed_url_state, model, indx],
            outputs=[],
            api_visibility="private",
        )

        button_base.click(
            fn=run,
            inputs=[
                aud,
                model,
                algo,
                algo_lvl,
                indx,
                indx_inf,
                res_fc,
                envel_r,
                const,
                denoise_gui,
                effects_gui,
                format_out,
                steps_gui,
            ],
            outputs=[output_base],
        ).success(
            fn=clear_player,
            inputs=None,
            outputs=[player_audio],
            queue=False,
            api_visibility="private",
        ).success(
            fn=load_first_audio,
            inputs=[output_base, player_gui],
            outputs=[player_audio],
            api_visibility="private",
        )

        gr.Examples(
            examples=[
                [
                    ["./test.ogg"],
                    "./model.pth",
                    "rmvpe+",
                    0,
                    "./model.index",
                    0.75,
                    3,
                    0.25,
                    0.50,
                ],
                [
                    ["./example2/test2.ogg"],
                    "./example2/model_link.txt",
                    "rmvpe+",
                    0,
                    "./example2/index_link.txt",
                    0.75,
                    3,
                    0.25,
                    0.50,
                ],
                [
                    ["./example3/test3.wav"],
                    "./example3/zip_link.txt",
                    "rmvpe+",
                    0,
                    None,
                    0.75,
                    3,
                    0.25,
                    0.50,
                ],
            ],
            fn=run,
            inputs=[
                aud,
                model,
                algo,
                algo_lvl,
                indx,
                indx_inf,
                res_fc,
                envel_r,
                const,
            ],
            outputs=[output_base],
            cache_examples=False,
        )
        gr.Markdown(RESOURCES)

    return app


if __name__ == "__main__":
    try:
        tts_voice_list = asyncio.new_event_loop().run_until_complete(get_voices_list(proxy=None))
        voices = sorted([
            (" - ".join(reversed(v["FriendlyName"].split("-"))).replace("Microsoft ", "").replace("Online (Natural)", f"({v['Gender']})").strip(), f"{v['ShortName']}-{v['Gender']}")
            for v in tts_voice_list
        ])
    except Exception as e:
        print(f"Warning: Could not retrieve online voices ({e}). Using default preset.")
        voices = [("English (United States) - EmmaMultilingual (Female)", "en-US-EmmaMultilingualNeural-Female")]
    
    app = get_gui()

    app.queue(default_concurrency_limit=40)

    app.launch(
        theme=selected_theme,
        css=APP_CSS,
        max_threads=40,
        share=IS_COLAB,
        show_error=True,
        quiet=False,
        debug=IS_COLAB,
        ssr_mode=False,
    )
