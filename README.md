---
title: RVC⚡ZERO
emoji: ⚡
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 5.43.1
app_file: app.py
license: mit
pinned: true
short_description: Voice conversion framework based on VITS
---

# RVC⚡ZERO

## Overview
**RVC Zero** is powered by the **Retrieval-based Voice Conversion (RVC)** framework. It supports both **voice conversion** and **text-to-speech (TTS)**, enabling you to input text or audio and transform it into expressive, high-quality outputs with control over voice characteristics and audio effects.

| Description         | Link |
|---------------------|------|
| 🎉 Repository       | [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/R3gm/rvc_zero_ui) |
| 🚀 Online DEMO      | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/r3gm/rvc_zero) |

## Key Functions
1. **Voice Conversion**  
   - Apply RVC to your own audio: upload via drag-and-drop, file selection, or use a URL for model, index, or audio files.  
   - Advanced settings allow fine-grained control over pitch, envelope, breath protection, and more.

2. **TTS (Text-to-Speech)**  
   - Enter text and synthesize speech using available voices (e.g., *English (United States) – Emma (Multilingual, Female)*).  
   - Preview playback with “Play,” then generate the voice output via “Process TTS.”

3. **Advanced Audio Controls**  
   Customize output with parameters like:  
   - **Pitch algorithm** (e.g., *rmvpe+*), **pitch level**, **index influence**,  
   - **Respiration median filtering**, **envelope ratio**, **consonant breath protection**,  
   - **Number of processing steps**, **output format** (e.g., WAV), **denoise**, **reverb**.

4. **Model Upload**  
   - Provide custom RVC model and index files via drag-and-drop or file picker to tailor conversion.

5. **Playback & Download**  
   - Listen to results within the interface and download processed TTS or converted audio outputs.

## Summary Table

| Function                    | Description                                                                   |
|-----------------------------|-------------------------------------------------------------------------------|
| **TTS Input & Playback**    | Type or paste text and generate synthetic speech in a selected voice.        |
| **Audio & Model Upload**    | Upload input audio, RVC model, and index files through intuitive UI.         |
| **Voice Conversion Control**| Adjust pitch, index influence, breath protection, and more for fine tuning.  |
| **Output Processing**       | Apply denoise, reverb, and choose output format for high-quality results.     |
| **Output Playback & Download** | Play synthesized audio or save the converted audio to your device.       |

