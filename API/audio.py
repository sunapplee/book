# Источник: REA/Preparing_Rea2026123/Module 4/API/audio.py
import numpy as np
import librosa
from sklearn import metrics
import joblib

def extract_features_mfcc(
    file_path,
    n_mfcc=30,
    n_fft=2048,
    hop_length=512,
    add_delta=True
):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)
    features = list(mfcc_mean) + list(mfcc_std)
    if add_delta:
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)
        delta_std  = np.std(delta, axis=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        delta2_mean = np.mean(delta2, axis=1)
        delta2_std  = np.std(delta2, axis=1)
        features += list(delta_mean) + list(delta_std)
        features += list(delta2_mean) + list(delta2_std)
    return np.array(features)

def update_label_encoder(encoder, new_labels):
    new_unique = np.unique(new_labels)
    old_unique = encoder.classes_
    added = [cls for cls in new_unique if cls not in old_unique]
    if added:
        print("Добавлены новые классы:", added)
        encoder.classes_ = np.concatenate([old_unique, added])
    return encoder

def inference_audio(new_audio_path):
    scaler = joblib.load("audio_models/scaler.pkl")
    encoder = joblib.load("audio_models/encoder.pkl")
    model = joblib.load("audio_models/audio_model.pkl")
    feats = extract_features_mfcc(new_audio_path[0])
    feats = scaler.transform([feats])
    y_pred = model.predict(feats)
    import logging
    logging.error(y_pred)
    response = encoder.inverse_transform([int(y_pred[0])])
    return response

def fine_tuning_audio(new_audio_paths, new_labels) -> dict:
    scaler = joblib.load("audio_models/scaler.pkl")
    encoder = joblib.load("audio_models/encoder.pkl")
    old_X = joblib.load("audio_models/old_X_audio.pkl")
    old_y = joblib.load("audio_models/old_y_audio.pkl")
    model = joblib.load("audio_models/audio_model.pkl")
    encoder = update_label_encoder(encoder, new_labels)
    joblib.dump(encoder, "audio_models/encoder.pkl")
    X_new = []
    for path in new_audio_paths:
        feats = extract_features_mfcc(path)
        X_new.append(feats)
    X_new = np.array(X_new)
    y_new = encoder.transform(new_labels)
    X_new_scaled = scaler.transform(X_new)
    X_full = np.vstack([old_X, X_new_scaled])
    y_full = np.concatenate([old_y, y_new])
    model.fit(X_full, y_full)
    X_test = joblib.load("audio_models/X_test_audio.pkl")
    y_test = joblib.load("audio_models/y_test_audio.pkl")
    y_pred = model.predict(X_test)
    metrics_dict = {
        "accuracy": round(metrics.accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(metrics.f1_score(y_test, y_pred, average="macro"), 4),
        "precision_macro": round(metrics.precision_score(y_test, y_pred, average="macro"), 4),
        "recall_macro": round(metrics.recall_score(y_test, y_pred, average="macro"), 4),
    }
    joblib.dump(model, "audio_models/audio_model.pkl")
    joblib.dump(X_full, "audio_models/old_X_audio.pkl")
    joblib.dump(y_full, "audio_models/old_y_audio.pkl")
    return metrics_dict
