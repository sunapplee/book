import requests

url = 'http://127.0.0.1:8660/'

print('-' * 100)
response = requests.get(url)
print(response.json())

# print('-' * 100)
# text = "Moscow is the capital of Russia and one of the largest cities in Europe."
# response = requests.get(url + 'ner',
#                         params={'text': text})
# print(response.json())


# print('-' * 100)
files = []
file_paths = ['../text1.pdf', '../text2.pdf']
for filepath in file_paths: 
    with open(filepath, 'rb') as f:
        files.append(('files', (filepath, f.read())))

response = requests.post(url + 'glm_ocr', files=files)
print(response.json())


# print('-' * 100)
# files = []
# file_paths = ['../text1.pdf', '../text2.pdf']
# for filepath in file_paths: 
#     with open(filepath, 'rb') as f:
#         files.append(('files', (filepath, f.read())))

# response = requests.post(url + 'tesseract_ocr', files=files)
# print(response.json())

print('-' * 100)
files = []
file_paths = ['../text1.pdf', '../text2.pdf']
for filepath in file_paths: 
    with open(filepath, 'rb') as f:
        files.append(('files', (filepath, f.read())))

response = requests.post(url + 'yolo_layout', files=files)
print(response.json())