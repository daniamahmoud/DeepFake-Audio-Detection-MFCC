import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

# ONNX imports
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort


def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

        # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio_data, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length
    )

    return np.mean(mfccs.T, axis=0)


def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))

    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")

    print("Number of samples in", directory, ":", len(X))
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    return X, y


def train_model(X, y):
    unique_classes = np.unique(y)
    print("Unique classes in y_train:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError("At least 2 samples per class are required to train")

    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("Insufficient samples for stratified splitting. Training on all data.")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Create pipeline (scaler + classifier)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', probability=False))
    ])

    pipeline.fit(X_train, y_train)

    # If test set exists, evaluate
    if X_test is not None:
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_mtx = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(conf_mtx)

    # --- Convert to ONNX ---
    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    # Save ONNX file
    onnx_filename = "mfcc_audio_model.onnx"
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model exported to {onnx_filename}")


def analyze_audio(input_audio_path):
    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")
        return
    if not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")
        return

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is None:
        print("Error: Failed to extract MFCC from audio.")
        return

    # Load ONNX model
    session = ort.InferenceSession("mfcc_audio_model.onnx")

    input_name = session.get_inputs()[0].name
    features = mfcc_features.reshape(1, -1).astype(np.float32)

    prediction = session.run(None, {input_name: features})[0]
    pred_class = int(prediction[0])

    if pred_class == 0:
        print("The input audio is classified as genuine.")
    else:
        print("The input audio is classified as deepfake.")


def main():
    genuine_dir = r"real_audio/"
    deepfake_dir = r"deepfake_audio/"

    X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
    X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)

    X = np.vstack((X_genuine, X_deepfake))
    y = np.hstack((y_genuine, y_deepfake))

    train_model(X, y)


if __name__ == "__main__":
    main()
#    user_input_file = "deepfake_audio/notme.wav"
#    analyze_audio(user_input_file)
