### requirements

- requirements/scripts.md
  — Подготовка к соревнованию (день 0): swap-файл (150–200 GB), переключение GPU (prime-select), алиасы bash
  — Настройка JupyterLab: LSP-автодополнение, конфиг, горячие клавиши
  — Загрузка файлов из GitHub через wget / curl
  — Пошаговая установка: обновление системы, базовые пакеты, Python, VS Code (+ расширения), Docker, Ollama
  — Создание виртуальных окружений: rea (general), pytorch_env, unsloth_env; активация и деактивация

- requirements/scripts.ipynb
  — Jupyter-версия инструкций по установке

- requirements/requirements-new.txt
  — Зависимости для полного окружения: CV, audio, NLP, geo, API, OCR, RAG, табличные данные

- requirements/requirements-general.txt
  — Зависимости для общего окружения (ML, геоданные, CV, web-разработка)

- requirements/requirements-torch.txt
  — Зависимости для PyTorch (deep learning, CUDA, vision, метрики)

- requirements/requirements-unsloth.txt
  — Зависимости для дообучения LLM моделей (Unsloth + LoRA)

### docker

- docker/docker.md
  — Работа с Docker
  — Работа с scp

### ollama

- ollama/ollama_rag.md
  — Пошаговая сборка RAG-пайплайна на LangChain

- ollama/ollama.md
  — Установка Ollama, управление моделями
  — Запуск текстовых, VLM и embedding-моделей
  — Поддержка моделей из HuggingFace в GGUF формате

- ollama/Modelfile
  — Конфигурация для сборки собственной модели в формате Ollama

- ollama/ollama_finetuning.py
  — Подготовка данных и дообучение Qwen3 (4-bit) с помощью Unsloth + LoRA

### images

- images/image_classic.py
  — Загрузка датасета: по папкам или по меткам в имени
  — Предобработка изображений (resize, normalize, RGB)
  — Балансировка классов через WeightedRandomSampler
  — Аугментации: flips, rotation, jitter, erasing, grayscale; расширение датасета
  — Train/test split со стратификацией
  — Обучение моделей: RandomForest, LogisticRegression, YOLOv8-cls

- images/image_deep.py
  — Подготовка данных для нейросетей (ImageFolder, трансформации)
  — Ручные признаки: гистограммы / уменьшение + flatten
  — Дообучение ResNet18
  — Fine-tuning и полное переобучение модели

- images/video.py
  — Извлечение кадров из видео
  — Классификация через YOLO-CLS
  — Аугментации: grayscale → RGB, color jitter, cutout

- images/yolo_clf.py
  — Разбиение датасета на train/val (80/20) через split_classify_dataset
  — Дообучение yolo26n-cls на задаче классификации
  — Инференс на валидации, метрики: Accuracy, F1-score
  — Визуализация: 9 случайных сэмплов с предсказанным и истинным классом

- images/yolo_det.py
  — Конвертация разметки из JSON (PyTorch) в YOLO txt-формат (xyxy → xywh, нормализация)
  — Разбиение датасета на train/val, конфигурационный data.yaml
  — Дообучение yolo26n на задаче детекции
  — Метрики: Precision, Recall, IoU, Dice (сопоставление bbox через IoU > 0.5)
  — Визуализация: предсказанные и истинные bbox на 4 случайных изображениях

- images/yolo_seg.py
  — Сегментация изображений «с учителем» на основе YOLO26n-seg
  — Подготовка данных: выравнивание вложенной структуры папок, бинаризация масок
  — Конвертация масок в YOLO-формат (convert_segment_masks_to_yolo_seg), split 80/10/10
  — Дообучение yolo26n-seg, метрики IoU и Dice, визуализация предсказаний

- images/yolo_tracking.py
  — Трекинг объектов на видео: YOLO26 + встроенный трекер
  — Конфигурация трекера, сохранение результата, конвертация в MP4
  — Построение траекторий движения объектов
  — Фильтрация трекинга по выбранным классам

- images/torch_clf.py
  — Классификация изображений на PyTorch: ResNet50, EfficientNet-B0, ConvNext-Small
  — Предобработка: resize, нормализация гистограммы; аугментация датасета
  — Метрики: Accuracy, F1-score на тестовой выборке

- images/torch_det.py
  — Детекция объектов на спутниковых снимках: Faster R-CNN, RetinaNet, SSD
  — Конвертация сегментационных масок в bounding boxes
  — Предобработка, аугментация, метрики: mAP, IoU, Recall

- images/torch_seg.py
  — Сегментация спутниковых снимков на PyTorch: UNet и другие архитектуры (smp)
  — Кастомный Dataset: загрузка изображений + бинарных масок, split 70/15/15
  — Обучение, метрики (IoU, Dice), визуализация предсказаний
  — Конвертация масок в YOLO-формат (convert_segment_masks_to_yolo_seg)

- images/segmentation.py
  — Сегментация изображений «без учителя»
  — Предобработка и аугментация данных
  — Формирование итогового расширенного датасета

- images/models.py
  — Универсальный пайплайн обучения: YOLO, ResNet18, SVM
  — Сохранение метрик и гиперпараметров в файл

- images/preprocess.py
  — Предобработка датасета: распаковка архива, resize 224×224, grayscale→RGB
  — Аугментации: поворот, добавление чёрного прямоугольника
  — Формирование train/test/valid структуры + zip-архив

### load_data

- load_data/load_data.py
  — Загрузка CSV, TXT, Excel, включая ленивую обработку через chunksize
  — Загрузка бинарных форматов с кастомной структурой (magic, headers, dtype)
  — Ленивая загрузка изображений через Path.iterdir() / Path.glob(); BytesIO для бинарных данных
  — Битый CSV: on_bad_lines="skip" и построчный поиск проблемных строк через csv.reader
  — JSON / JSONL: pd.read_json, построчный orjson, потоковый ijson
  — PDF: извлечение текста и изображений через pdfplumber
  — Строковые литералы: безопасный разбор через ast.literal_eval

- load_data/load_multimodal.py
  — Парсинг кастомного RSA-формата построчно, оптимизация типов (float64→float32), EDA + визуализация
  — Загрузка изображений через PIL, cv2, torchvision, skimage, bytes
  — Загрузка npz-аудио, конвертация в wav/mp3/m4a; чтение через librosa, torchaudio, soundfile

### text

- text/text.py
  — Преобразование текста в числовые признаки (TF-IDF)

- text/text_qa_roberta.py
  — Fine-tuning RoBERTa для Question Answering на SberQuAD (извлечение ответа из контекста)
  — Токенизация с выравниванием позиций ответа, HuggingFace Trainer
  — Подбор гиперпараметров, метрики Exact Match и F1
  — Сравнение с базовой RoBERTa, сохранение модели

- text/text_ner.py
  — Fine-tuning DistilBERT для NER на датасете WNUT17 (токенная классификация)
  — Токенизация и выравнивание BIO-меток, DataCollatorForTokenClassification
  — Обучение через HuggingFace Trainer, метрики Precision/Recall/F1
  — Инференс на новых текстах через pipeline

- text/text_clustering_routing.py
  — Кластеризация вопросов (SberQuAD): KMeans, Agglomerative, BERTopic; метод локтя, дендрограмма
  — Эмбеддинги через Ollama, визуализация: PCA, UMAP; метрики Silhouette/DB/CH
  — Маршрутизация текста: fine-tuning ruBERT для многоклассовой классификации по кластерам
  — Анализ частотности слов + облако слов (WordCloud)

- text/text_qa_bm25.py
  — Предобработка QA-пар: очистка, стоп-слова, стемминг (joblib.Parallel)
  — Аугментация вопросов: Word-level и Char-level (augmentex)
  — QA-поиск через BM25Okapi (rank_bm25)
  — Оценка качества: эмбеддинги Qwen3-Embedding (Ollama) + косинусное сходство

- text/text_matching.py
  — Предобработка: исправление латиница/кириллица, emoji, стоп-слова, стемминг (joblib.Parallel)
  — Метрики схожести строк: Hamming, Jaro-Winkler, Levenshtein, Jaccard, Tanimoto, Prefix/Postfix
  — Косинусная схожесть эмбеддингов: LaBSE, RuBERT-tiny2, FRIDA, USER
  — Классическая векторизация: CountVectorizer, TF-IDF
  — Fine-tuning RuBERT (DeepPavlov) для бинарного matching пар текстов; метрики Acc/F1/AUC

### table

- table/table.py
  — Сохранение таблиц в БД SQLite (to_sql)
  — Хранение нереляционных данных через shelve
  — Боксплоты для визуализации
  — Анализ плотности, нормальности (Шапиро-Уилк), skew/kurtosis

- table/table_clustering.py
  — Определение оптимального числа кластеров (метод локтя)
  — Кластеризация: KMeans, HDBSCAN, BIRCH
  — PCA-визуализация кластеров
  — Метрики качества кластеризации (Silhouette, CH, DB)

- table/table_dashboard.py
  — Интерактивный Streamlit-дашборд
  — Фильтры по колонкам и диапазону дат
  — Метрики и визуализация загрязнителей (линии, бары, scatter)

- table/table_feature_importance.py
  — chunksize для больших данных
  — Анализ признаков и распределений
  — Корреляционная матрица
  — SHAP-анализ
  — Permutation Importance
  — Оптимизация типов данных + сохранение в parquet (макс. сжатие)

- table/table_time_series.py
  — Интерполяция пропусков
  — STL-декомпозиция: тренд, сезонность, остаток
  — Прогнозирование: простая ML-модель + SARIMAX

- table/table_kaggle_clf.py
  — Kaggle Tabular Playground: бинарная классификация (Heart Disease, ROC AUC)
  — EDA: распределения, выбросы, корреляции, дисбаланс классов
  — Baseline → генерация и отбор признаков (PolynomialFeatures, Permutation Importance, SHAP)
  — Сравнение алгоритмов: CatBoost, LightGBM, XGBoost, RandomForest, LogisticRegression
  — Подбор гиперпараметров через Optuna
  — Смешивание моделей (blending)

- table/table3.py
  — Подбор гиперпараметров (RandomizedSearchCV)
  — Балансировка классов через SMOTE
  — ROC-AUC для мульткласса
  — Сохранение моделей (CatBoost, MLP, scaler)
  — Fine-tuning регрессионных моделей

### audio

- audio/audio.py
  — Вычисление аудио-статистик (длительность, громкость, вариативность)
  — Визуализация: временной сигнал, STFT-спектрограмма, Mel-спектрограмма
  — Извлечение признаков: MFCC, chroma, centroid, bandwidth, rolloff (mean/std)

- audio/audio2.py
  — Визуализация waveform
  — Разрезание аудио на сегменты + фильтрация тишины
  — Формирование датасета + train/val/test
  Представления аудио:
  — Waveform
  — Mel-спектрограммы (2D)
  — MFCC признаки
  Дополнительно:
  — Обучение Logistic Regression на MFCC
  — Корреляции MFCC и гистограммы по классам

- audio/audio3.py
  — Извлечение признаков: MFCC
  — Подбор гиперпараметров (GridSearchCV)
  — Fine-tuning аудио-классификатора

- audio/audio_mel_clf.py
  — Предобработка: шумоподавление (noisereduce), VAD (Silero)
  — Извлечение признаков: MFCC, Delta, Delta-Delta, Chroma, Spectral Contrast/Centroid/Bandwidth/Rolloff
  — ML-бэйслайн: CatBoost на табличных фичах
  — Аугментации: Time Stretch, Pitch Shift, Time/Freq Masking, Add Noise, Volume Gain
  — Генерация Mel-спектрограмм → обучение ConvNeXt-Tiny; сравнение подходов

- audio/audio_command_clf.py
  — Разметка аудио в Label Studio; загрузка и векторизация через Wav2Vec2
  — Baseline: LogisticRegression на эмбеддингах, метрики WER/CER
  — Предобработка: шумоподавление (DeepFilterNet), VAD (Silero), фильтрация тишины
  — Транскрибация Whisper large-v3 + нечёткое сопоставление команд (rapidfuzz)

### geo

- geo/geo_gpx.py
  — Работа с GPX-треками: точки, LineString, проекции
  — Визуализация на OSM + генерация изображений
  — Извлечение окружения через OSMnx

- geo/geo_shp_tif.py
  — Анализ векторных геоданных (SHP): породы, полигоны, визуализация
  — Обработка растров (TIF): чтение, выравнивание, аугментации
  — Интерактивные карты (OSM, спутник)

- geo/geo_image.py
  — Чтение HDF5 спутниковых снимков
  — Чтение band_stats, label_map, partition
  — Расчёт индексов NDVI и NDMI + визуализация
  — Формирование набора: 13 каналов + NDVI + NDMI = 15 каналов
  — PyTorch Dataset + DataLoader

- geo/geo.py
  — Карта с предсказаниями (геовизуализация классификации)

- geo/geo_features.py
  — H3: преобразование координат в гексагональные ячейки, агрегаты по ячейкам
  — Haversine: расстояние по поверхности Земли как ML-признак
  — Shapely: принадлежность точки полигону, буферные зоны

### API

- API/multimodal_ml_api.py
	— FastAPI-сервис для инференса и дообучения ML-моделей
	— Поддержка табличных, аудио, image и geo-данных
- API/multimodal_ml_test_api.py
	— Тестирование API-запросов к сервису

- API/doc_processing_api.py
	— FastAPI-сервис для обработки PDF-документов
	— OCR через Tesseract и GLM-OCR
	— Layout Detection через YOLOv10 (doclayout_yolo)
	— NER через HuggingFace pipeline
- API/doc_processing_test_api.py
	— Тестирование API-запросов к сервису
	— Проверка OCR и Layout Detection на PDF-документах
	— Отправка тестовых файлов и вывод результатов в консоль

- API/field_segmentation_api.py
	— FastAPI-сервис для сегментации полей на спутниковых изображениях
	— Загрузка, предобработка, сегментация и визуализация масок
	— Расчёт площади выделенных участков по сегментации
- API/test_field_segmentation_api.py
	— Тестирование API-запросов к сервису

- API/audio_command_api.py
	— FastAPI-сервис для обработки и распознавания голосовых команд
	— Шумоподавление, детекция речи, фильтрация и транскрибация аудио
	— Поддержка метрик качества распознавания и изменения конфигурации аудио

### ocr

- ocr/ocr_comparison_1.py
  — Сравнение OCR-решений для PDF: Pytesseract, Marker-PDF (Surya), Docling, Nanonets-OCR
  — Конвертация PDF в изображения (pdf2image), замер скорости каждого метода

- ocr/ocr_comparison_2.py
  — Сравнение OCR-решений для PDF: PP-OCRv5, GLM-OCR, DeepSeek OCR
  — Замер скорости и качества распознавания

- ocr/ocr_layout_detection.py
  — Сравнение методов layout-анализа PDF: Surya, PP-DocLayout, YOLO-based, LayoutLMv3, Detectron2
  — Детекция блоков (текст, таблицы, изображения), оценка скорости и качества

### telegram_bot

- telegram_bot/tg_bot.py
  — Телеграм-бот на aiogram, принимающий фото и текстовые команды и возвращающий результаты классификации через удобный чат-интерфейс

### streamlit

- streamlit/multimodal_ml_gui.py
	— Streamlit-интерфейс для работы с мультимодальными данными
	— Инференс и дообучение tabular, audio, image и geo моделей
	— Загрузка файлов, ввод параметров и отображение результатов
	— Визуализация геоданных и прогнозов на интерактивной карте

- streamlit/doc_processing_gui.py
	— Streamlit-интерфейс для загрузки PDF-документов
	— Запуск OCR, Layout Detection и NER через API
	— Выбор OCR-модели: Tesseract или GLM-OCR
	— Просмотр структуры документа и результатов распознавания

- streamlit/field_segmentation_gui.py
	— Streamlit-интерфейс для загрузки и анализа изображений
	— Запуск сегментации, просмотр масок и контуров полей
	— Отображение площади найденных участков
	
- streamlit/audio_command_gui.py
	— Streamlit-интерфейс для загрузки и анализа аудиофайлов
	— Визуализация сигнала, шумоподавление и выделение речи
	— Распознавание команд и оценка качества результатов

### docs

- docs/user_doc_link.md
  — Шаблон руководства (ссылка)

- docs/instruction_simple.docx
	— Руководство по эксплуатации (простая версия)

- docs/user_instruction.docx
	— Руководство по эксплуатации (базовая версия)
  
- docs/pres_filled.pptx
	— Заполненная презентация
  
- docs/pres_nonfilled.pptx
	— Презентация (шаблон)

- docs/pres_link.md
	— Презентация (ссылка на шаблон)

- docs/api_docs.md
  — Документация REST API

### justifications

- justifications.md
  — Обоснование выбора float32 для оптимизации памяти
  — Выбор типов графиков для EDA: lineplot, boxplot, histplot
  — Анализ выбросов и поведения сигналов по группам признаков
  — Выбор алгоритма сегментации без учителя: MiniBatch K-means vs Hierarchical vs SLIC (теорема Клейнберга)
  — Выбор архитектур сегментации: U-Net, FPN, DeepLab V3, LR-ASPP, YOLO26n-seg
  — Обоснование split 70/15/15 (train/val/test)
  — Выбор моделей регрессии: LinearRegression, DecisionTree, LightGBM (bias–variance трейдофф)
