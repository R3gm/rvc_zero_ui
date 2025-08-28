import os
import gradio as gr
import spaces
from infer_rvc_python import BaseLoader
import random
import logging
import time
import soundfile as sf
from infer_rvc_python.main import download_manager, load_hu_bert, Config
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
import shutil
import threading
import argparse
import sys

parser = argparse.ArgumentParser(description="Run the app with optional sharing")
parser.add_argument(
    '--share',
    action='store_true',
    help='Enable sharing mode'
)
parser.add_argument(
    '--theme',
    type=str,
    default="aliabid94/new-theme",
    help='Set the theme (default: aliabid94/new-theme)'
)
args = parser.parse_args()

IS_COLAB = True if ('google.colab' in sys.modules or args.share) else False
IS_ZERO_GPU = os.getenv("SPACES_ZERO_GPU")

logging.getLogger("infer_rvc_python").setLevel(logging.ERROR)

converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)
converter.hu_bert_model = load_hu_bert(Config(only_cpu=False), converter.hubert_path)

test_model = "https://huggingface.co/sail-rvc/Aldeano_Minecraft__RVC_V2_-_500_Epochs_/resolve/main/model.pth?download=true, https://huggingface.co/sail-rvc/Aldeano_Minecraft__RVC_V2_-_500_Epochs_/resolve/main/model.index?download=true"
test_names = ["model.pth", "model.index"]

for url, filename in zip(test_model.split(", "), test_names):
    try:
        download_manager(
            url=url,
            path=".",
            extension="",
            overwrite=False,
            progress=True,
        )
        if not os.path.isfile(filename):
            raise FileNotFoundError
    except Exception:
        with open(filename, "wb") as f:
            pass

title = "<center><strong><font size='7'>RVC⚡ZERO</font></strong></center>"
description = "This demo is provided for educational and research purposes only. The authors and contributors of this project do not endorse or encourage any misuse or unethical use of this software. Any use of this software for purposes other than those intended is solely at the user's own risk. The authors and contributors shall not be held responsible for any damages or liabilities arising from the use of this demo inappropriately." if IS_ZERO_GPU else ""
theme = args.theme
delete_cache_time = (3200, 3200) if IS_ZERO_GPU else (86400, 86400)

PITCH_ALGO_OPT = [
    "pm",
    "harvest",
    "crepe",
    "rmvpe",
    "rmvpe+",
]


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
        return a_, b_

    txt_files = []
    for base_file in [a_, b_]:
        if base_file is not None and base_file.endswith(".txt"):
            txt_files.append(base_file)

    directory = os.path.dirname(a_)

    for txt in txt_files:
        with open(txt, 'r') as file:
            first_line = file.readline()

        download_manager(
            url=first_line.strip(),
            path=directory,
            extension="",
        )

    for f in find_files(directory):
        if f.endswith(".zip"):
            unzip_in_folder(f, directory)

    model = None
    index = None
    end_files = find_files(directory)

    for ff in end_files:
        if ff.endswith(".pth"):
            model = os.path.join(directory, ff)
            gr.Info(f"Model found: {ff}")
        if ff.endswith(".index"):
            index = os.path.join(directory, ff)
            gr.Info(f"Index found: {ff}")

    if not model:
        gr.Error(f"Model not found in: {end_files}")

    if not index:
        gr.Warning("Index not found")

    return model, index


def ensure_valid_file(url):
    if "huggingface" not in url:
        raise ValueError("Only downloads from Hugging Face are allowed")

    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            content_length = response.headers.get("Content-Length")

        if content_length is None:
            raise ValueError("No Content-Length header found")

        file_size = int(content_length)
        if file_size > 900000000 and IS_ZERO_GPU:
            raise ValueError("The file is too large. Max allowed is 900 MB.")

        return file_size

    except Exception as e:
        raise e


def clear_files(directory):
    time.sleep(15)
    print(f"Clearing files: {directory}.")
    shutil.rmtree(directory)


def get_my_model(url_data, progress=gr.Progress(track_tqdm=True)):

    if not url_data:
        return None, None

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
            download_manager(
                url=link,
                path=directory,
                extension="",
            )

        for f in find_files(directory):
            if f.endswith(".zip"):
                unzip_in_folder(f, directory)

        model = None
        index = None
        end_files = find_files(directory)

        for ff in end_files:
            if ff.endswith(".pth"):
                model = ff
                gr.Info(f"Model found: {ff}")
            if ff.endswith(".index"):
                index = ff
                gr.Info(f"Index found: {ff}")

        if not model:
            raise ValueError(f"Model not found in: {end_files}")

        if not index:
            gr.Warning("Index not found")
        else:
            index = os.path.abspath(index)

        return os.path.abspath(model), index

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
def convert_now(audio_files, random_tag, converter, type_output, steps):
    for step in range(steps):
        audio_files = converter(
            audio_files,
            random_tag,
            overwrite=False,
            parallel_workers=(2 if IS_COLAB else 8),
            type_output=type_output,
        )

    return audio_files


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
        raise ValueError("The audio pls")

    if isinstance(audio_files, str):
        audio_files = [audio_files]

    try:
        duration_base = librosa.get_duration(filename=audio_files[0])
        print("Duration:", duration_base)
    except Exception as e:
        print(e)

    if file_m is not None and file_m.endswith(".txt"):
        file_m, file_index = find_my_model(file_m, file_index)
        print(file_m, file_index)

    random_tag = "USER_"+str(random.randint(10000000, 99999999))

    converter.apply_conf(
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
    time.sleep(0.1)

    result = convert_now(audio_files, random_tag, converter, type_output, steps)

    if active_noise_reduce:
        result = apply_noisereduce(result, type_output)

    if audio_effects:
        result = add_audio_effects(result, type_output)

    return result


def audio_conf():
    return gr.File(
        label="Audio files",
        file_count="multiple",
        type="filepath",
        container=True,
    )


def model_conf():
    return gr.File(
        label="Model file",
        type="filepath",
        height=130,
    )


def pitch_algo_conf():
    return gr.Dropdown(
        PITCH_ALGO_OPT,
        value=PITCH_ALGO_OPT[4],
        label="Pitch algorithm",
        visible=True,
        interactive=True,
    )


def pitch_lvl_conf():
    return gr.Slider(
        label="Pitch level",
        minimum=-24,
        maximum=24,
        step=1,
        value=0,
        visible=True,
        interactive=True,
    )


def index_conf():
    return gr.File(
        label="Index file",
        type="filepath",
        height=130,
    )


def index_inf_conf():
    return gr.Slider(
        minimum=0,
        maximum=1,
        label="Index influence",
        value=0.75,
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


def button_conf():
    return gr.Button(
        "Inference",
        variant="primary",
    )


def output_conf():
    return gr.File(
        label="Result",
        file_count="multiple",
        interactive=False,
    )


def active_tts_conf():
    return gr.Checkbox(
        False,
        label="TTS",
        # info="",
        container=False,
    )


def tts_voice_conf():
    return gr.Dropdown(
        label="tts voice",
        choices=voices,
        visible=False,
        value="en-US-EmmaMultilingualNeural-Female",
    )


def tts_text_conf():
    return gr.Textbox(
        value="",
        placeholder="Write the text here...",
        label="Text",
        visible=False,
        lines=3,
    )


def tts_button_conf():
    return gr.Button(
        "Process TTS",
        variant="secondary",
        visible=False,
    )


def tts_play_conf():
    return gr.Checkbox(
        False,
        label="Play",
        # info="",
        container=False,
        visible=False,
    )


def sound_gui():
    return gr.Audio(
        value=None,
        type="filepath",
        # format="mp3",
        autoplay=True,
        visible=True,
        interactive=False,
        elem_id="audio_tts",
    )


def steps_conf():
    return gr.Slider(
        minimum=1,
        maximum=3,
        label="Steps",
        value=1,
        step=1,
        interactive=True,
    )


def format_output_gui():
    return gr.Dropdown(
        label="Format output:",
        choices=["wav", "mp3", "flac"],
        value="wav",
    )

def denoise_conf():
    return gr.Checkbox(
        False,
        label="Denoise",
        # info="",
        container=False,
        visible=True,
    )


def effects_conf():
    return gr.Checkbox(
        False,
        label="Reverb",
        # info="",
        container=False,
        visible=True,
    )


def infer_tts_audio(tts_voice, tts_text, play_tts):
    out_dir = "output"
    folder_tts = "USER_"+str(random.randint(10000, 99999))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, folder_tts), exist_ok=True)
    out_path = os.path.join(out_dir, folder_tts, "tts.mp3")

    asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(out_path))
    if play_tts:
        return [out_path], out_path
    return [out_path], None


def show_components_tts(value_active):
    return gr.update(
        visible=value_active
    ), gr.update(
        visible=value_active
    ), gr.update(
        visible=value_active
    ), gr.update(
        visible=value_active
    )


def down_active_conf():
    return gr.Checkbox(
        False,
        label="URL-to-Model",
        # info="",
        container=False,
    )


def down_url_conf():
    return gr.Textbox(
        value="",
        placeholder="Write the url here...",
        label="Enter URL",
        visible=False,
        lines=1,
    )


def down_button_conf():
    return gr.Button(
        "Process",
        variant="secondary",
        visible=False,
    )


def show_components_down(value_active):
    return gr.update(
        visible=value_active
    ), gr.update(
        visible=value_active
    ), gr.update(
        visible=value_active
    )

CSS = """
#audio_tts {
  visibility: hidden;   /* invisible but still takes space */
  height: 0px;
  width: 0px;
  max-width: 0px;
  max-height: 0px;
}
"""

def get_gui(theme):
    with gr.Blocks(theme=theme, css=CSS, fill_width=True, fill_height=False, delete_cache=delete_cache_time) as app:
        gr.Markdown(title)
        gr.Markdown(description)

        active_tts = active_tts_conf()
        with gr.Row():
            with gr.Column(scale=1):
                tts_text = tts_text_conf()
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            tts_voice = tts_voice_conf()
                            tts_active_play = tts_play_conf()

                tts_button = tts_button_conf()
                tts_play = sound_gui()

        active_tts.change(
            fn=show_components_tts,
            inputs=[active_tts],
            outputs=[tts_voice, tts_text, tts_button, tts_active_play],
        )

        aud = audio_conf()
        # gr.HTML("<hr>")

        tts_button.click(
            fn=infer_tts_audio,
            inputs=[tts_voice, tts_text, tts_active_play],
            outputs=[aud, tts_play],
        )

        down_active_gui = down_active_conf()
        down_info = gr.Markdown(
            f"Provide a link to a zip file, like this one: `https://huggingface.co/MrDawg/ToothBrushing/resolve/main/ToothBrushing.zip?download=true`, or separate links with a comma for the .pth and .index files, like this: `{test_model}`",
            visible=False
        )
        with gr.Row():
            with gr.Column(scale=3):
                down_url_gui = down_url_conf()
            with gr.Column(scale=1):
                down_button_gui = down_button_conf()

        with gr.Column():
            with gr.Row():
                model = model_conf()
                indx = index_conf()

        down_active_gui.change(
            show_components_down,
            [down_active_gui],
            [down_info, down_url_gui, down_button_gui]
        )

        down_button_gui.click(
            get_my_model,
            [down_url_gui],
            [model, indx]
        )

        with gr.Accordion(label="Advanced settings", open=False):
            algo = pitch_algo_conf()
            algo_lvl = pitch_lvl_conf()
            indx_inf = index_inf_conf()
            res_fc = respiration_filter_conf()
            envel_r = envelope_ratio_conf()
            const = consonant_protec_conf()
            steps_gui = steps_conf()
            format_out = format_output_gui()
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        denoise_gui = denoise_conf()
                        effects_gui = effects_conf()
        button_base = button_conf()
        output_base = output_conf()

        button_base.click(
            run,
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

    return app


if __name__ == "__main__":
    tts_voice_list = asyncio.new_event_loop().run_until_complete(get_voices_list(proxy=None))
    voices = sorted([
        (" - ".join(reversed(v["FriendlyName"].split("-"))).replace("Microsoft ", "").replace("Online (Natural)", f"({v['Gender']})").strip(), f"{v['ShortName']}-{v['Gender']}")
        for v in tts_voice_list
    ])

    app = get_gui(theme)

    app.queue(default_concurrency_limit=40)

    app.launch(
        max_threads=40,
        share=IS_COLAB,
        show_error=True,
        quiet=False,
        debug=IS_COLAB,
        ssr_mode=False,
    )
