import streamlit as st
import requests


st.title('Система распознавания документов')

url = 'http://127.0.0.1:8660/'

files = st.file_uploader('Загрузите документ(ы)', accept_multiple_files=True, type=['pdf'])

if files or st.session_state.get('files', []):
    st.session_state['files'] = files

    if files:
        st.session_state['files_bytes'] = [
            (file.name, file.read()) for file in files
        ]

    files2api = [
        ('files', (name, data))
        for name, data in st.session_state.get('files_bytes', [])
    ]


    if st.button('Обнаружение структуры документа') or st.session_state.get('layout', []):

        if st.session_state.get('layout', []):
            page_num = st.slider(label='Выберите страницу', max_value=len(st.session_state['layout']), min_value=0, step=1)
            st.image(st.session_state['layout'][page_num], width=300)
        else:
            response = requests.post(url + 'yolo_layout', files=files2api)
            st.session_state['layout'] = response.json()['paths']

            page_num = st.slider(label='Выберите страницу', max_value=len(st.session_state['layout']), min_value=0, step=1)

            st.image(st.session_state['layout'][page_num], width=300)

    model = st.selectbox(
    "Выбор модели?",
    ("Быстрая (tesseract)", "Качественная (GLM-OCR)"),
    )

    if st.button('Очистить'):
            st.session_state['ner_ocr'] = st.session_state['ocr']
            st.session_state['ocr'] = ' '

    if st.button('Обнаружить текст') or st.session_state.get('ocr', []):

        if st.session_state.get('ocr', []):
            st.markdown(st.session_state['ocr'])
        
        else:
            if model == "Быстрая (tesseract)":
                response = requests.post(url + 'tesseract_ocr', files=files2api)
                st.session_state['ocr'] = response.json()['result']

                st.markdown(st.session_state['ocr'])

            elif model == "Качественная (GLM-OCR)":
                response = requests.post(url + 'glm_ocr', files=files2api)
                st.session_state['ocr'] = response.json()['result']

                st.markdown(st.session_state['ocr'])

    if st.button('Обнаружить сущности') or st.session_state.get('ner', []):

        if st.session_state.get('ner', []):
            st.json(st.session_state['ner'])
        else:
            st.session_state['ner'] = requests.get(url + 'ner',
                         params={'text': st.session_state['ner_ocr'][:1000]}).json()['result']
            st.write("NER TEXT:", st.session_state.get('ner_ocr', '')[:200])
            st.json(st.session_state['ner'])
            

