import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import librosa
from moviepy import VideoFileClip
import tempfile
import joblib
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import subprocess
import requests
import base64

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Configuration
SPOTIFY_CLIENT_ID="9d2b7e6e01b54e348bf05ccae1af6175"
SPOTIFY_CLIENT_SECRET="1e3b71b912574094b5c634616c9a101d"
MODEL_PATH = "multimodal_3dcnn.pth"
PREPROCESSOR_PATH = "preprocessor.pkl"
SPOTIFY_DATA_PATH = "dataset.csv"
UPLOAD_FOLDER = "Uploads"
ALLOWED_EXTENSIONS = {'mp4', 'wav', 'mp3', 'webm'}
NUM_FRAMES = 16
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load label encoder
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    le = preprocessor['label_encoder']
    print(f"Loaded label encoder with classes: {le.classes_}")
except Exception as e:
    print(f"Error loading preprocessor: {e}")
    raise

# Load Spotify dataset
try:
    spotify_data = pd.read_csv(SPOTIFY_DATA_PATH)
except Exception as e:
    print(f"Error loading Spotify dataset: {e}")
    raise

# Define transforms for audio
audio_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define Multimodal 3D CNN model
class Multimodal3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Multimodal3DCNN, self).__init__()
        
        # Video branch (3D CNN)
        self.video_cnn = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.AdaptiveAvgPool3d((2, 8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Audio branch (2D CNN)
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Combined dense layers
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, video, audio):
        print(f"Input video shape: {video.shape}")
        video = video.permute(0, 2, 1, 3, 4)
        print(f"Permuted video shape: {video.shape}")
        video_features = self.video_cnn(video)
        audio_features = self.audio_cnn(audio)
        combined = torch.cat((video_features, audio_features), dim=1)
        output = self.fc(combined)
        return output

# Load the trained model
try:
    model = Multimodal3DCNN(num_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Spotify API authentication
def get_spotify_access_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_bytes = auth_string.encode('ascii')
    auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
    headers = {
        'Authorization': f'Basic {auth_b64}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {'grant_type': 'client_credentials'}
    try:
        response = requests.post(auth_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()['access_token']
    except Exception as e:
        print(f"Error getting Spotify access token: {e}")
        raise

# Spotify search function
def search_spotify(query):
    access_token = get_spotify_access_token()
    search_url = 'https://api.spotify.com/v1/search'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'q': query, 'type': 'track', 'limit': 10}
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        tracks = response.json()['tracks']['items']
        return [
            {
                'track_id': track['id'],
                'track_name': track['name'],
                'artists': ', '.join(artist['name'] for artist in track['artists'])
            }
            for track in tracks
        ]
    except Exception as e:
        print(f"Error searching Spotify: {e}")
        return []

# Helper functions for preprocessing (unchanged)
def extract_random_faces(video_path, max_faces=16, frame_size=(64, 64), sample_frames=20):
    faces = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return faces

    sample_frames = min(sample_frames, total_frames)
    frame_indices = sorted(random.sample(range(total_frames), sample_frames))

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces_detected) > 0:
            (x, y, w, h) = max(faces_detected, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = x*2, y*2, w*2, h*2
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, frame_size)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            faces.append(face)
        if len(faces) >= max_faces:
            break

    cap.release()
    if not faces:
        print(f"No faces detected in {video_path}.")
    return faces

def pad_or_truncate_frames(faces, target_num_frames=16, frame_shape=(64, 64, 3)):
    faces_array = np.array(faces)
    current_num_frames = len(faces_array)
    
    if current_num_frames == 0:
        return np.zeros((target_num_frames, *frame_shape), dtype=np.uint8)
    
    if current_num_frames > target_num_frames:
        return faces_array[:target_num_frames]
    elif current_num_frames < target_num_frames:
        padding = np.repeat(faces_array[-1:][np.newaxis], target_num_frames - current_num_frames, axis=0)
        return np.concatenate([faces_array, padding], axis=0)
    else:
        return faces_array

def extract_mel_spectrogram(file_path, n_mels=128, max_len=128):
    try:
        if file_path.endswith('.mp4'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
                audio_path = temp_audio_file.name
            video = VideoFileClip(file_path)
            if video.audio is None:
                video.close()
                return None
            video.audio.write_audiofile(audio_path)
            video.close()
        else:
            audio_path = file_path

        y, sr = librosa.load(audio_path, sr=22050)
        if file_path.endswith('.mp4'):
            os.remove(audio_path)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < max_len:
            pad_width = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]

        mel_db = np.stack([mel_db, mel_db, mel_db], axis=-1)
        return mel_db
    except Exception as e:
        print(f"Error processing audio for {file_path}: {e}")
        return None

def preprocess_video(video_path):
    faces = extract_random_faces(video_path, max_faces=NUM_FRAMES, sample_frames=20)
    if not faces:
        return None
    
    video_np = pad_or_truncate_frames(faces, target_num_frames=NUM_FRAMES)
    video_np = video_np.astype(np.float32) / 255.0
    video_np = (video_np - 0.5) / 0.5
    video_tensor = torch.tensor(video_np, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    print(f"Video tensor shape: {video_tensor.shape}")
    return video_tensor

def preprocess_audio(audio_path):
    mel = extract_mel_spectrogram(audio_path, n_mels=128, max_len=128)
    if mel is None:
        return None
    audio_tensor = audio_transform(mel).unsqueeze(0).to(device)
    print(f"Audio tensor shape: {audio_tensor.shape}")
    return audio_tensor

def predict_from_tensor(video_tensor, audio_tensor, model, label_encoder):
    model.eval()
    with torch.no_grad():
        output = model(video_tensor, audio_tensor)
        _, predicted = torch.max(output, 1)
        emotion = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    return emotion

# Music recommendation (unchanged)
features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness']
scaler = MinMaxScaler()
spotify_features = scaler.fit_transform(spotify_data[features])

def get_emotion_profile(emotion):
    emotion_profiles = {
        'happy': {'danceability': 0.7, 'energy': 0.6, 'valence': 0.8, 'tempo': 120, 'loudness': -5, 'acousticness': 0.3},
        'sad': {'danceability': 0.4, 'energy': 0.2, 'valence': 0.2, 'tempo': 70, 'loudness': -15, 'acousticness': 0.7},
        'angry': {'danceability': 0.5, 'energy': 0.8, 'valence': 0.3, 'tempo': 140, 'loudness': -3, 'acousticness': 0.2},
        'calm': {'danceability': 0.3, 'energy': 0.1, 'valence': 0.5, 'tempo': 60, 'loudness': -20, 'acousticness': 0.9}
    }
    return pd.DataFrame([emotion_profiles.get(emotion, emotion_profiles['calm'])], columns=features)

def get_opposite_emotion(predicted_emotion):
    opposite_emotions = {
        'happy': 'sad',
        'sad': 'happy',
        'angry': 'calm',
        'calm': 'angry'
    }
    return opposite_emotions.get(predicted_emotion, 'happy')

def recommend_music(predicted_emotion, spotify_data, spotify_features, total_n=30, mood_lifting_ratio=0.5):
    emotion_profile = get_emotion_profile(predicted_emotion)
    emotion_features = scaler.transform(emotion_profile)

    similarity_scores = cosine_similarity(emotion_features, spotify_features)[0]
    spotify_data['similarity'] = similarity_scores

    num_fitting = int(total_n * (1 - mood_lifting_ratio))
    top_fitting = spotify_data.sort_values(by='similarity', ascending=False).head(num_fitting * 2)
    mood_fitting = top_fitting.sample(n=num_fitting)

    opposite_emotion = get_opposite_emotion(predicted_emotion)
    opposite_profile = get_emotion_profile(opposite_emotion)
    opposite_features = scaler.transform(opposite_profile)

    opposite_similarity_scores = cosine_similarity(opposite_features, spotify_features)[0]
    spotify_data['opposite_similarity'] = opposite_similarity_scores

    num_lifting = total_n - num_fitting
    top_lifting = spotify_data.sort_values(by='opposite_similarity', ascending=False).head(num_lifting * 2)
    mood_lifting = top_lifting.sample(n=num_lifting)

    recommendations = pd.concat([mood_fitting, mood_lifting]).drop_duplicates(subset=['track_name', 'artists'])
    if len(recommendations) < total_n:
        additional = spotify_data.sample(n=total_n - len(recommendations))
        recommendations = pd.concat([recommendations, additional]).drop_duplicates(subset=['track_name', 'artists']).head(total_n)

    recommendations['recommendation_type'] = ['mood-fitting'] * len(mood_fitting) + ['mood-lifting'] * len(mood_lifting)
    return {
        'mood_fitting': recommendations[recommendations['recommendation_type'] == 'mood-fitting'][['track_id', 'track_name', 'artists', 'similarity', 'recommendation_type']],
        'mood_lifting': recommendations[recommendations['recommendation_type'] == 'mood-lifting'][['track_id', 'track_name', 'artists', 'opposite_similarity', 'recommendation_type']]
    }

# Flask API endpoints
@app.route('/predict', methods=['POST'])
def predict():
    print("Request received")
    if 'file' not in request.files:
        print("Missing 'file' field")
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    print(f"Received file: {file.filename}")

    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Allowed formats: mp4, wav, mp3, webm'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        file_ext = filename.rsplit('.', 1)[1].lower()

        if file_ext == 'webm':
            try:
                print("Converting .webm to .mp4 using ffmpeg...")
                mp4_path = file_path.replace('.webm', '.mp4')
                cmd = [
                    'ffmpeg',
                    '-i', file_path,
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    mp4_path
                ]
                subprocess.run(cmd, check=True)
                os.remove(file_path)
                file_path = mp4_path
                file_ext = 'mp4'
                print(f"Converted to MP4: {file_path}")
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg conversion failed: {e}")
                return jsonify({'error': 'FFmpeg conversion failed'}), 500

        # Process input based on file type
        if file_ext == 'mp4':
            video_tensor = preprocess_video(file_path)
            audio_tensor = preprocess_audio(file_path)
            if video_tensor is None or audio_tensor is None:
                os.remove(file_path)
                return jsonify({'error': 'Failed to process video or audio'}), 400
        elif file_ext in ['wav', 'mp3']:
            video_tensor = torch.zeros(1, 16, 3, 64, 64).to(device)
            audio_tensor = preprocess_audio(file_path)
            if audio_tensor is None:
                os.remove(file_path)
                return jsonify({'error': 'Failed to process audio'}), 400
        else:
            os.remove(file_path)
            return jsonify({'error': 'Unsupported file format'}), 400

        # Predict emotion
        emotion = predict_from_tensor(video_tensor, audio_tensor, model, le)

        # Get music recommendations
        recommendations = recommend_music(emotion, spotify_data, spotify_features, total_n=30, mood_lifting_ratio=0.5)
        
        # Format response
        response = {
            'predicted_emotion': emotion,
            'mood_fitting_recommendations': recommendations['mood_fitting'][['track_id', 'track_name', 'artists', 'similarity']].to_dict(orient='records'),
            'mood_lifting_recommendations': recommendations['mood_lifting'][['track_id', 'track_name', 'artists', 'opposite_similarity']].to_dict(orient='records')
        }

        # Clean up
        os.remove(file_path)
        return jsonify(response), 200

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    try:
        results = search_spotify(query)
        return jsonify({'tracks': results}), 200
    except Exception as e:
        return jsonify({'error': f'Error searching Spotify: {str(e)}'}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)