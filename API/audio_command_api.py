from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Literal
from contextlib import asynccontextmanager
import base64
import io

import noisereduce as nr
from df import enhance, init_df
from pyrnnoise import RNNoise
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import numpy as np
from faster_whisper import WhisperModel
import torch
import soundfile as sf
import librosa
from jiwer import cer, wer

import uvicorn

import re
import time



class AudioNoiseReduce(BaseModel):
    filename: str
    reducer: Literal['df', 'nr', 'rnnoise']

class AudioPath(BaseModel):
    filename: str

class AudioFilter(BaseModel):
    filename: str
    timestamps: list

class AudioTranscribe(BaseModel):
    filename: str
    model: Literal['large_v3', 'tiny', 'turbo']

class AudioResample(BaseModel):
    filename: str
    sr: int 
    mono: bool


class AudioMetrics(BaseModel):
    filename: str
    reference: str

lifespan_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Сохраняем моделей
    lifespan_data['df_model'], lifespan_data['df_state'], _ = init_df()  # Load default model

    lifespan_data['denoiser_RNNoise'] = RNNoise(sample_rate=48000)

    lifespan_data['vad_model'] = load_silero_vad()

    # Инициализируем модель
    lifespan_data['whisper_model_large'] = WhisperModel("large-v3", device="cuda", compute_type="float16")
    lifespan_data['whisper_model_turbo'] = WhisperModel("turbo", device="cuda", compute_type="float16")
    lifespan_data['whisper_model_tiny'] = WhisperModel("tiny", device="cuda", compute_type="float16")

    lifespan_data['command'] = {}
    lifespan_data['time'] = {}

    yield
    # Отчистка ресурсов
    lifespan_data.clear()


app = FastAPI(lifespan=lifespan)


@app.post('/load_data')
async def load_data(request: Request):
    # Обработка списка аудио
    form = await request.form()
    audios = []
    for data in form:
        contents = form[data].file.read()
        filename = form[data].filename
        # Сохраняем
        with open(f'original_files/{filename}', 'wb') as f:
            f.write(contents)

        # кодируем в base64
        audio_base64 = base64.b64encode(contents).decode()

        audios.append({
            "filename": filename,
            "audio": audio_base64
        })

    # Возвращаем названия файлов
    return audios


@app.post('/reduce_noise')
async def reduce_noise(audio: AudioNoiseReduce):


    if audio.reducer == 'df':
        # Загружаем аудио
        loaded_audio, sr = librosa.load(f'original_files/{audio.filename}', sr=None)
        loaded_audio = torch.tensor(loaded_audio).unsqueeze(dim=0)
        reduced_noise = enhance(lifespan_data['df_model'], lifespan_data['df_state'], loaded_audio)
        reduced_noise = reduced_noise.squeeze().cpu().numpy()
        sf.write(f"denoised_audio/{audio.filename}", reduced_noise, sr)
    elif audio.reducer == 'nr':
        # Загружаем аудио
        loaded_audio, sr = librosa.load(f'original_files/{audio.filename}', sr=16000)
        reduced_noise = nr.reduce_noise(loaded_audio, sr)
        sf.write(f"denoised_audio/{audio.filename}", reduced_noise, 16000)
    else:
        # Обрабатываем аудиофайл и сохраняем результат
        for speech_prob in lifespan_data['denoiser_RNNoise'].denoise_wav(f"original_files/{audio.filename}", f"denoised_audio/{audio.filename}"):
            print(f"Processing frame with speech probability: {speech_prob}")
        reduced_noise, sr = librosa.load( f"denoised_audio/{audio.filename}", sr=None)

    # переводим numpy → wav bytes
    buffer = io.BytesIO()
    sf.write(buffer, reduced_noise, sr, format="WAV")
    buffer.seek(0)

    audio_base64 = base64.b64encode(buffer.read()).decode()

    return {'filename': audio.filename, 'audio': audio_base64}


@app.post("/resample_audio")
async def resample_audio(audio: AudioResample):

    loaded_audio, sr = librosa.load(
        f"original_files/{audio.filename}",
        sr=None,
        mono=audio.mono
    )
    # resample
    resampled_audio = librosa.resample(
        loaded_audio,
        orig_sr=sr,
        target_sr=audio.sr
    )
    sf.write(
        f"original_files/{audio.filename}",
        resampled_audio,
        audio.sr
    )

    return {"filename": audio.filename}


@app.post('/detect_speech')
async def detect_speech(audio: AudioPath):
    loaded_audio, sr = librosa.load(
        f"denoised_audio/{audio.filename}",
        sr=None,
    )

    # Silero ожидает torch.Tensor
    wav = torch.from_numpy(loaded_audio).unsqueeze(0)

    # Получаем тайминги голосовых сегментов
    speech_timestamps = get_speech_timestamps(
      wav,
      lifespan_data['vad_model'],
      return_seconds=False,
    )

    buffer = io.BytesIO()
    sf.write(buffer, loaded_audio, sr, format="WAV")
    buffer.seek(0)

    audio_base64 = base64.b64encode(buffer.read()).decode()

    # Возвращаем тайминги
    return {'filename': audio.filename,
            'timestamps': speech_timestamps,
            'audio': audio_base64}


@app.post('/filter_audio')
async def filter_audio(audio: AudioFilter):

    loaded_audio, sr = librosa.load(
        f"denoised_audio/{audio.filename}",
        sr=None,
    )

    # Сохраняем сюда фрагменты речи
    fragments = []

    # Сохраняем аудио на позициях start:end
    for ts in audio.timestamps:
        s, e = ts["start"], ts["end"]
        fragments.append(loaded_audio[s:e])

    new_audio = np.concatenate(fragments)

    sf.write(
        f"vad_audios/{audio.filename}",
        new_audio,
        sr
    )

    buffer = io.BytesIO()
    sf.write(buffer, new_audio, sr, format="WAV")
    buffer.seek(0)

    audio_base64 = base64.b64encode(buffer.read()).decode()

    # Возвращаем тайминги
    return {'filename': audio.filename,
            'audio': audio_base64}
    

@app.post('/get_command')
async def get_command(audio: AudioTranscribe):

    if audio.model == 'large_v3':
        model = lifespan_data['whisper_model_large']
    elif audio.model == 'turbo':
        model = lifespan_data['whisper_model_turbo']
    elif audio.model == 'tiny':
        model = lifespan_data['whisper_model_tiny'] 

    st_time = time.time()
    segments, info = model.transcribe(f'vad_audios/{audio.filename}',
                                     language='ru',
                                     temperature=0, )
    
    # Собираем фрагменты в одну строку
    text = " ".join([segment.text for segment in segments])

    text = text.lower().strip()
    text = re.sub(r'[^A-Za-zА-Яа-яЁё0-9 ]', '', text)

    end_time = time.time()

    lifespan_data['command'][audio.filename] = text
    lifespan_data['time'][audio.filename] = end_time - st_time

    return {'command': text,
            'filename': audio.filename}


@app.post('/get_metrics')
async def get_metrics(audio: AudioMetrics):

    pred_command = lifespan_data['command'][audio.filename]
    time_pipeline = lifespan_data['time'][audio.filename]

    wer_command = wer(pred_command, audio.reference)
    cer_command = cer(pred_command, audio.reference)

    return {
        'time': time_pipeline,
        'wer': round(wer_command, 3),
        'cer': round(cer_command, 3)
    }


if __name__ == '__main__':
    uvicorn.run("api:app", host='0.0.0.0', port=8900, reload=True)
