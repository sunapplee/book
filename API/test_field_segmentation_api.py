import requests
from PIL import Image
import io

url = 'http://127.0.0.1:8080/'


file_paths = [r'E:\Heckfy\atom\Atomskills2026\15.02.2026\homework\preprocess_images\images\image_image_00001_20210420_const_0004\image_image_00001_20210420_const_0004_brightning.jpg',
r'E:\Heckfy\atom\Atomskills2026\19.02.2026\homework\valid_new\images\image_image_01008_20210720_const_0001.jpg']


# answer = requests.post(url + 'load_images',
#                         files=[
#                             ('file_1', open(file_paths[0], 'rb')),
#                             ('file_2', open(file_paths[1], 'rb')),
#                         ])
# print(answer.json())
# answer = requests.post(
#     url + 'preprocess_image',
#     json={'filename': 'image_image_01008_20210720_const_0001.jpg'}
# )
# print(answer.json())

# answer = requests.post(
#     url + 'segment_image',
#     json={'filename': 'image_image_01008_20210720_const_0001.jpg'}
# )

# mask = Image.open(io.BytesIO(answer.content))

# print(mask)
# mask.show()


# answer = requests.get(
#     url + 'visualize_image',
#     params={'image_filename': 'image_image_01008_20210720_const_0001.jpg'}
# )

# img = Image.open(io.BytesIO(answer.content))

# img.show()



answer = requests.get(
    url + 'get_area',
    params={'image_filename': 'image_image_01008_20210720_const_0001.jpg'}
)

print(answer.json())