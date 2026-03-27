import streamlit as st
import requests
import base64
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt

url = 'http://127.0.0.1:8900/'

st.title('Веб-интерфейс для определения команд')

if "noise_removed" not in st.session_state:
    st.session_state.noise_removed = False

if "speech_detected" not in st.session_state:
    st.session_state.speech_detected = False

if "speech_cut" not in st.session_state:
    st.session_state.speech_cut = False


if "sr_mono" not in st.session_state:
    st.session_state.sr_mono = False

if "transcribe" not in st.session_state:
    st.session_state.transcribe = False

files = st.file_uploader(
    'Загрузите файл(ы)',
    accept_multiple_files=True,
    type=['mp3', 'wav']
)

if files:

    st.success('Данные загружены')

    files_to_api = []
    for file in files:
        bytes_file = file.getvalue()
    
        files_to_api.append((file.name, bytes_file))
    
    answer = requests.post(url + 'load_data',
                    files=files_to_api)
    
    st.session_state.audio_files = answer.json()
    if st.session_state.audio_files:
        for af in st.session_state.audio_files:
            # st.write(af['filename'])
            audio_bytes = base64.b64decode(af['audio'])
            st.audio(audio_bytes, format="audio/wav")
            audio_np, sr = sf.read(io.BytesIO(audio_bytes))

            fig, ax = plt.subplots()
            librosa.display.waveshow(ax=ax, y=audio_np, )
            ax.set_title(af['filename'])
            st.pyplot(fig)

        # st.json(st.session_state.audio_files)


    type_noise_remover = st.selectbox(
        'Метод удаления шума',
        options=['NoiseReduce', 'DeepFilterNet', 'RNNoise']
    )

    cols = st.columns(2)

    with cols[0]:

        if st.button('Убрать шум'):
            st.session_state.noise_removed = True

            for audio in st.session_state.audio_files:

                reducer = {'NoiseReduce': 'nr',
                           'DeepFilterNet': 'df',
                           'RNNoise': 'rnnoise'}[type_noise_remover]

                answer = requests.post(url + 'reduce_noise',
                                    json={'filename': audio['filename'],
                                          'reducer': reducer})
                
                audio_reduce = answer.json()
                audio_bytes = base64.b64decode(audio_reduce['audio'])
                st.audio(audio_bytes, format="audio/wav")
                audio_np, sr = sf.read(io.BytesIO(audio_bytes))
                fig, ax = plt.subplots()
                librosa.display.waveshow(ax=ax, y=audio_np, )
                ax.set_title(audio['filename'])
                st.pyplot(fig)

            st.success('Шум удален!')

        if st.session_state.noise_removed:

            if st.button('Определить речь'):
                st.session_state.speech_detected = True
                st.session_state.timestamps = {}
                for audio in st.session_state.audio_files:
                    st.write(audio['filename'])

                    answer = requests.post(url + 'detect_speech',
                                        json={'filename': audio['filename']})
                    audio_vad = answer.json()
                    audio_bytes = base64.b64decode(audio_vad['audio'])
                    audio_np, sr = sf.read(io.BytesIO(audio_bytes))

                    coords = [tuple(i.values()) for i in audio_vad['timestamps']]

                    # Определяем данные для визуализации
                    stamps = np.arange(len(audio_np))
                    condlist = [
                        (stamps >= pair[0]) & (stamps <= pair[1])
                        for pair in coords
                    ]
                    choices = [True] * len(coords)
                    step = np.select(condlist, choicelist=choices, default=False)
                    fig, ax = plt.subplots()
                    ax.plot(audio_np)
                    ax.plot(step, color="red") 
                    st.pyplot(fig)

                    st.session_state.timestamps[audio['filename']] = audio_vad['timestamps']

                st.success('Найдена речь!')

        if st.session_state.speech_detected:

            if st.button('Оставить только речь'):
                st.session_state.speech_cut = True
                for audio in st.session_state.audio_files:

                    answer = requests.post(url + 'filter_audio',
                                            json={'filename': audio['filename'],
                                                  'timestamps': st.session_state.timestamps[audio['filename']]})
                    audio_filter = answer.json()
                    st.write(audio['filename'])
                    st.audio(base64.b64decode(audio_filter['audio']), format="audio/wav")


                st.success('Аудио очищено!')

        if st.session_state.speech_cut:
            whisper_model = st.selectbox(
                'Модель распознавания',
                options=['Tiny', 'Large', 'Turbo']
                )

            if st.button('Определить команду'):
                st.session_state.transcribe = True

                model2api = {'Large': 'large_v3', 
                            'Tiny': 'tiny', 
                            'Turbo': 'turbo'}[whisper_model]
                
                for audio in st.session_state.audio_files:
                        answer = requests.post(url + 'get_command',
                                                json={'filename': audio['filename'],
                                                        'model': model2api})
                        audio_transcribe = answer.json()
                        st.markdown(f"```{audio_transcribe['filename']}```: **{audio_transcribe['command']}**")

            if st.session_state.transcribe:
                
                references = {}

                for audio in st.session_state.audio_files:

                    ref = st.text_input(
                        f"Референс для {audio['filename']}",
                        key=f"ref_{audio['filename']}"
                    )

                    references[audio['filename']] = ref
                if st.button("Получить метрику"):
                    results = []

                    for audio in st.session_state.audio_files:

                        ref = references[audio['filename']]

                        answer = requests.post(
                            url + 'get_metrics',
                            json={
                                "filename": audio['filename'],
                                "reference": ref
                            }
                        )

                        metric = answer.json()
                        results.append({
                        "filename": audio["filename"],
                        "time": metric["time"],
                        "wer": metric["wer"],
                        "cer": metric["cer"]
                        })
                    df = pd.DataFrame(results)
                    st.dataframe(df)
    with cols[1]:

        st.subheader('Изменить конфигурацию')

        sr_new = st.number_input(
            label='Частота дискретизации',
            min_value=1000,
            max_value=50000,
            step=1000,
            value=16000
        )

        mono = st.selectbox(
            label='Mono/Stereo',
            options=['Mono', 'Stereo']
        )

        mono2api = mono == 'Mono'

        if st.button('Изменить конфигурацию'):
            for audio in st.session_state.audio_files:

                answer = requests.post(
                    url + 'resample_audio',
                    json={
                        "filename": audio['filename'],
                        "mono": mono2api,
                        'sr': sr_new
                    }
                )


            st.success('Успешно!')