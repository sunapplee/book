import streamlit as st
import io
import requests
from PIL import Image

# Конфигурация
url = 'http://127.0.0.1:8080/'



st.title('Определение полей на изображении')
st.markdown('#### Интерфейс использует методы и функции из API предыдущего модуля по адресу ```localhost:8000```')

files = st.file_uploader('Загрузите одно или несколько изображений, формат JPG',
                         accept_multiple_files=True, type="jpg")



seg_button = st.button('Сегментировать поля')

if seg_button:
    if not files:
        st.warning('Пожалуйста, загрузите изображение')
    else:
        files_to_api = []
        for file in files:
            bytes_file = file.getvalue()
        
            files_to_api.append((file.name, bytes_file))
        
        answer = requests.post(url + 'load_images',
                        files=files_to_api)
        
        filenames = answer.json()['file_names']
        if answer.status_code == 200:
            st.success('Изображения загружены!')
            print(answer.json())
            for filename in filenames:
                answer = requests.post(
                url + 'preprocess_image',
                json={'filename': filename}
            )
            st.success('Изображения предобработаны!')

            masks = []
            for filename in filenames:
                answer = requests.post(
                url + 'segment_image',
                json={'filename': filename}
            )
                mask = Image.open(io.BytesIO(answer.content))
                masks.append(mask)
            st.success('Сегментация прошла успешно')


            images = []
            for filename in filenames:
                answer = requests.get(
                url + 'visualize_image',
                params={'image_filename': filename}
            )
                img = Image.open(io.BytesIO(answer.content))
                images.append(img)


            areas = []
            for filename in filenames:
                answer = requests.get(
                url + 'get_area',
                params={'image_filename': filename}
            )
                area = answer.json()
                areas.append(area)

            st.subheader('Результаты сегментации')

            for name, img_, mask_, area_ in zip(filenames, images, masks, areas):
                cols = st.columns(3)
                with cols[0]:
                    st.markdown('#### Площадь поля')
                    if not area_['result']:
                        st.error('Предупреждение! Полей не обнаружено!')
                    for cluster in area_['result']:
                        st.markdown(f"Кластер `{cluster['claster']}`: *{cluster['area']}* га")

                with cols[1]:
                    st.markdown('#### Маска сегментации')
                    st.image(mask_)
                with cols[2]:
                    st.markdown('#### Контуры полей')
                    st.image(img_)


        else:
            st.error('Изображения загружены с ошибкой')



    