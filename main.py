import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from tempfile import NamedTemporaryFile
import shutil
import numpy as np
import pandas as pd
import librosa
import nltk
import speech_recognition as sr
import traceback
from typing import Dict, Any, List, Optional
import uvicorn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import uuid
import joblib
import re
from nltk.metrics.distance import edit_distance
from nltk.corpus import wordnet
import json
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import speech_recognition as speech_rec
from pandas.api.types import is_categorical_dtype
from pandas.api.types import CategoricalDtype

app = FastAPI(
    title="MemoTag Speech Intelligence API",
    description="API for analyzing speech patterns to detect cognitive decline markers",
    version="1.0.0"
)
# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global storage for batch analysis (in-memory only)
analysis_results = {}
batch_analysis_results = {}

# Download NLTK resources on startup
@app.on_event("startup")
def download_resources():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

def extract_audio_features(audio_path: str) -> Dict[str, float]:
    """Extract acoustic features from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract pauses
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        pauses = len(non_silent_intervals) - 1
        pause_duration = duration - sum([(end - start) / sr for start, end in non_silent_intervals])
        pause_rate = pauses / duration if duration > 0 else 0
        
        # Extract pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches_nonzero = pitches[pitches > 0]
        pitch_mean = np.mean(pitches_nonzero) if len(pitches_nonzero) > 0 else 0
        pitch_std = np.std(pitches_nonzero) if len(pitches_nonzero) > 0 else 0
        
        # Speech rate (acoustic)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        speech_rate_acoustic = len(onsets) / duration if duration > 0 else 0
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        
        # MFCC features (voice quality)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Voice stability - using pyin for better jitter calculation
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=75, fmax=600)
            f0_voiced = f0[voiced_flag]
            jitter = np.mean(np.abs(np.diff(f0_voiced))) if len(f0_voiced) > 1 else 0
        except Exception:
            # Fallback to the previous jitter calculation if pyin fails
            pitch_diff = np.diff(pitches_nonzero) if len(pitches_nonzero) > 1 else np.array([0])
            jitter = np.mean(np.abs(pitch_diff)) if len(pitch_diff) > 0 else 0
        
        # Energy variation (shimmer approximation)
        rms = librosa.feature.rms(y=y)[0]
        shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
        
        features = {
            'duration': duration,
            'pause_count': pauses,
            'pause_duration': pause_duration,
            'pause_rate': pause_rate,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'speech_rate_acoustic': speech_rate_acoustic,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'jitter': jitter,
            'shimmer': shimmer
        }
        
        # Add MFCC features
        for i, (mean, std) in enumerate(zip(mfcc_means, mfcc_stds)):
            features[f'mfcc_{i+1}_mean'] = mean
            features[f'mfcc_{i+1}_std'] = std
            
        return features
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        print(traceback.format_exc())
        return {
            'duration': 0, 'pause_count': 0, 'pause_duration': 0, 'pause_rate': 0,
            'pitch_mean': 0, 'pitch_std': 0, 'speech_rate_acoustic': 0,
            'spectral_centroid': 0, 'spectral_rolloff': 0, 'jitter': 0, 'shimmer': 0
        }
    
def preprocess_audio(audio_path: str) -> str:
    """Preprocess audio to improve speech recognition quality.
    Returns path to preprocessed audio file."""
    try:
        import librosa
        import soundfile as sf
        from scipy import signal
        import numpy as np
        import os
        
        # Create temp file for processed audio
        processed_path = f"{audio_path}_processed.wav"
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # 1. Noise reduction using spectral gating
        # Simple noise reduction approach
        def reduce_noise(y, sr):
            # Get a noise profile from the first 0.5 seconds (assuming it's noise/silence)
            noise_sample = y[:int(sr * 0.5)] if len(y) > int(sr * 0.5) else y
            noise_profile = np.mean(np.abs(noise_sample))
            
            # Simple noise gate
            y_reduced = np.copy(y)
            mask = np.abs(y) < noise_profile * 2  # Threshold at 2x noise level
            y_reduced[mask] = 0
            return y_reduced
            
        y = reduce_noise(y, sr)
        
        # 2. Normalization - adjust volume to optimal level
        y = librosa.util.normalize(y)
        
        # 3. Low-pass filter to remove high-frequency noise
        # Cut off at 3000Hz (most speech is below this)
        b, a = signal.butter(5, 3000/(sr/2), 'low')
        y = signal.filtfilt(b, a, y)
        
        # 4. Save processed audio
        sf.write(processed_path, y, sr)
        
        return processed_path
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        print(traceback.format_exc())
        return audio_path  # Return original if processing fails

def speech_to_text(audio_path: str) -> str:
    """Convert speech to text with multiple fallback options."""
    try:
        recognizer = speech_rec.Recognizer()
        with speech_rec.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            
            # Try Google first
            try:
                return recognizer.recognize_google(audio_data)
            except speech_rec.UnknownValueError:
                print(f"Google couldn't understand audio - trying Sphinx")
                
            # Fallback to CMU Sphinx
            try:
                return recognizer.recognize_sphinx(audio_data)
            except speech_rec.UnknownValueError:
                print(f"Sphinx couldn't understand audio")
                
            return ""
            
    except Exception as e:
        print(f"Error in speech_to_text: {e}")
        return ""
    
def detect_word_recall_issues(text: str) -> Dict[str, Any]:
    """Detect word recall issues like substitutions and word finding difficulties."""
    try:
        # First check if text is empty or None
        if not text:
            return {'substitution_score': 0, 'word_finding_difficulty': 0, 'incomplete_sentences': 0}
            
        # Tokenize text
        words = nltk.word_tokenize(text.lower())
        if not words:
            return {'substitution_score': 0, 'word_finding_difficulty': 0, 'incomplete_sentences': 0}
            
        # Rest of function remains the same
        # POS tagging
        pos_tags = nltk.pos_tag(words)
        
        # Calculate substitution score - detect semantically inappropriate words
        blob = TextBlob(text)
        sentences = blob.sentences
        
        # Look for semantic inconsistencies within sentences
        substitution_score = 0
        if sentences:
            for sentence in sentences:
                words = sentence.words
                if len(words) >= 4:  # Only analyze sentences with enough words
                    # Check if any word seems out of place semantically
                    for i, word in enumerate(words):
                        if i > 0 and i < len(words) - 1:
                            context = " ".join([str(words[i-1]), str(words[i+1])])
                            similarity = semantic_similarity(str(word), context)
                            if similarity < 0.2:  # Low similarity might indicate substitution
                                substitution_score += 1
        
        # Word finding difficulty - detect pauses before nouns, long pauses, repetitions
        word_finding_score = 0
        
        # Count repetitions of words
        repetitions = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repetitions += 1
                
        # Count instances where nouns are preceded by hesitation markers
        for i in range(1, len(pos_tags)):
            if pos_tags[i][1].startswith('NN') and pos_tags[i-1][0] in ['um', 'uh', 'er', 'ah']:
                word_finding_score += 1
        
        # Incomplete sentences (ending abruptly or trailing off)
        incomplete_sentences = 0
        for sentence in blob.sentences:
            sentence_str = str(sentence)
            # Check for sentences without proper ending punctuation or trailing off
            if not sentence_str.endswith(('.', '!', '?')) or sentence_str.endswith('...'):
                incomplete_sentences += 1
                
            # Check for sentences that end with determiners or prepositions
            last_word_pos = pos_tags[-1][1] if pos_tags else ""
            if last_word_pos in ['DT', 'IN', 'CC']:
                incomplete_sentences += 1
                
        substitution_score = min(1.0, substitution_score / max(1, len(sentences)))
        word_finding_difficulty = (word_finding_score + repetitions) / max(1, len(words))
        incomplete_sentences_ratio = incomplete_sentences / max(1, len(blob.sentences))
        
        return {
            'substitution_score': substitution_score,
            'word_finding_difficulty': word_finding_difficulty,
            'incomplete_sentences': incomplete_sentences_ratio
        }
    except Exception as e:
        print(f"Error detecting word recall issues: {e}")
        return {'substitution_score': 0, 'word_finding_difficulty': 0, 'incomplete_sentences': 0}



def semantic_similarity(word: str, context: str) -> float:
    """Calculate semantic similarity between a word and context."""
    # Simple implementation using WordNet when available
    try:
        # Get WordNet synsets
        word_synsets = wordnet.synsets(word)
        context_words = context.split()

        context_synsets = []
        for context_word in context_words:
            context_synsets.extend(wordnet.synsets(context_word))
            
        if not word_synsets or not context_synsets:
            return 0.5  # Neutral score if no synsets found
            
        # Calculate max similarity between any two synsets
        max_sim = 0
        for word_syn in word_synsets:
            for context_syn in context_synsets:
                try:
                    sim = word_syn.path_similarity(context_syn)
                    if sim and sim > max_sim:
                        max_sim = sim
                except:
                    continue
                    
        return max_sim if max_sim > 0 else 0.5
    except:
        return 0.5  # Default neutral score on errors

def extract_linguistic_features(text: str) -> Dict[str, float]:
    """Extract linguistic features from transcribed text."""
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        
        # Add null check at the beginning
        if not text:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_words_per_sentence': 0,
                'hesitation_rate': 0,
                'type_token_ratio': 0,
                'unique_word_ratio': 0,
                'filler_word_ratio': 0,
                'proper_noun_ratio': 0,
                'content_word_ratio': 0,
                'unique_content_ratio': 0,
                'coherence_score': 0,
                'sentence_completion': 0
            }
            
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Additional check after tokenization
        if not words or not sentences:
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_words_per_sentence': 0,
                'hesitation_rate': 0,
                'type_token_ratio': 0,
                'unique_word_ratio': 0,
                'filler_word_ratio': 0,
                'proper_noun_ratio': 0,
                'content_word_ratio': 0,
                'unique_content_ratio': 0,
                'coherence_score': 0,
                'sentence_completion': 0
            }
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Hesitation markers
        hesitation_markers = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'well']
        hesitation_count = sum(1 for word in words if word in hesitation_markers)
        hesitation_rate = hesitation_count / word_count if word_count > 0 else 0
        
        # Vocabulary diversity
        unique_words = set(words)
        type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0
        
        # Get stop words for content word analysis
        stop_words = set(stopwords.words('english'))
        content_words = [w for w in words if w not in stop_words and w.isalpha()]
        unique_content_words = set(content_words)
        
        # Filler words
        filler_words = ['thing', 'stuff', 'something', 'this', 'that', 'it']
        filler_count = sum(1 for word in words if word in filler_words)
        filler_word_ratio = filler_count / word_count if word_count > 0 else 0
        
        # Proper nouns
        proper_nouns = [w for w in words if len(w) > 0 and w[0].isupper() and w.isalpha()]
        proper_noun_ratio = len(proper_nouns) / word_count if word_count > 0 else 0
        
        # Coherence analysis (from your original code)
        coherence_score = calculate_coherence(sentences)
        
        # Analyze sentence completion (from your original code)
        sentence_completion = analyze_sentence_completion(text)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': avg_words_per_sentence,
            'hesitation_rate': hesitation_rate,
            'type_token_ratio': type_token_ratio,
            'unique_word_ratio': len(unique_words) / word_count if word_count > 0 else 0,
            'content_word_ratio': len(content_words) / word_count if word_count > 0 else 0,
            'unique_content_ratio': len(unique_content_words) / len(content_words) if content_words else 0,
            'filler_word_ratio': filler_word_ratio,
            'proper_noun_ratio': proper_noun_ratio,
            'coherence_score': coherence_score,
            'sentence_completion': sentence_completion
        }
    except Exception as e:
        print(f"Error extracting linguistic features: {e}")
        print(traceback.format_exc())
        return {
            'word_count': 0, 'sentence_count': 0, 'avg_words_per_sentence': 0,
            'hesitation_rate': 0, 'type_token_ratio': 0, 'unique_word_ratio': 0,
            'filler_word_ratio': 0, 'proper_noun_ratio': 0, 'content_word_ratio': 0,
            'unique_content_ratio': 0, 'coherence_score': 0, 'sentence_completion': 0
        }




def calculate_coherence(sentences) -> float:
    """Calculate text coherence based on sentence-to-sentence similarity."""
    if len(sentences) <= 1:
        return 1.0  # Default high score for single sentences
        
    try:
        # Convert sentences to vector representation (simple bag of words)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        
        try:
            sentence_vectors = vectorizer.fit_transform([str(s) for s in sentences])
            
            # Calculate cosine similarities between adjacent sentences
            similarities = []
            for i in range(len(sentences)-1):
                sim = cosine_similarity(
                    sentence_vectors[i:i+1],
                    sentence_vectors[i+1:i+2]
                )[0][0]
                similarities.append(sim)
                
            # Return average similarity
            return sum(similarities) / len(similarities) if similarities else 0.5
        except:
            return 0.5  # Return neutral score on errors
    except:
        return 0.5  # Return neutral score on errors

def analyze_sentence_completion(text: str) -> float:
    """Analyze sentence completion patterns."""
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0.0
            
        incomplete_count = 0
        for sentence in sentences:
            # Check for trailing off
            if sentence.endswith('...') or re.search(r'\b(um|uh|er|ah)\s*$', sentence):
                incomplete_count += 1
                continue
                
            # Check for sentences ending with prepositions or determiners
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            if pos_tags and pos_tags[-1][1] in ['IN', 'DT', 'CC']:
                incomplete_count += 1
                
        # Return completion rate (inverted incomplete rate)
        return 1.0 - (incomplete_count / len(sentences))
    except Exception as e:
        print(f"Error analyzing sentence completion: {e}")
        return 0.5  # Neutral score on error

def calculate_wpm(text: str, duration: float) -> float:
    """Calculate words per minute."""
    # Add null check
    if not text:
        return 0.0
    
    try:
        words = nltk.word_tokenize(text)
        wpm = (len(words) / duration * 60) if duration > 0 else 0
        return wpm
    except Exception as e:
        print(f"Error calculating WPM: {e}")
        return 0.0


def apply_unsupervised_ml(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Apply unsupervised ML techniques to detect anomalies with proper feature handling."""
    if features_df.empty or len(features_df) < 2:
        return {
            'anomaly_scores': [0.0],
            'clusters': [0],
            'is_anomaly': [False]
        }
        
    try:
        # Prepare feature matrix - select numerical columns and handle missing values
        feature_columns = features_df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove non-feature columns that shouldn't be part of the ML analysis
        exclude_columns = ['sample_id', 'risk_score', 'anomaly_score', 'is_anomaly', 'cluster']
        feature_columns = [col for col in feature_columns if col not in exclude_columns]
        
        if not feature_columns:
            print("Warning: No valid feature columns found for ML analysis")
            return {
                'anomaly_scores': [0.0] * len(features_df),
                'clusters': [0] * len(features_df),
                'is_anomaly': [False] * len(features_df)
            }
        
        # Extract feature matrix and fill missing values
        X = features_df[feature_columns].fillna(0).values
        
        # Replace infinite values with large numbers
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalize/standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save feature names used for future reference
        with open(f"model_feature_names.json", "w") as f:
            json.dump(feature_columns, f)
        
        # 1. Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.2, random_state=42, n_estimators=100)
        anomaly_scores = iso_forest.fit_predict(X_scaled)
        
        # Convert from 1 (normal) and -1 (anomaly) to anomaly score between 0-1
        anomaly_scores = np.where(anomaly_scores == 1, 0, 1)
        
        # 2. DBSCAN for clustering
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        clusters = dbscan.fit_predict(X_scaled)
        
        # 3. Determine anomalies (both from Isolation Forest and DBSCAN's -1 cluster)
        is_anomaly = np.logical_or(anomaly_scores == 1, clusters == -1)
        
        # Save model for future use
        model_path = "isolation_forest_model.joblib"
        joblib.dump(iso_forest, model_path)
        
        return {
            'anomaly_scores': anomaly_scores.tolist(),
            'clusters': clusters.tolist(),
            'is_anomaly': is_anomaly.tolist()
        }
    except Exception as e:
        print(f"Error applying unsupervised ML: {e}")
        print(traceback.format_exc())
        return {
            'anomaly_scores': [0.0] * len(features_df),
            'clusters': [0] * len(features_df),
            'is_anomaly': [False] * len(features_df)
        }

def generate_visualizations(features_df: pd.DataFrame, batch_id: str) -> Dict[str, str]:
    """Generate visualizations of key speech features."""
    try:
        # Create batch-specific directory first
        batch_dir = f"visualizations/{batch_id}"
        os.makedirs(batch_dir, exist_ok=True)
        
        visualizations = {}
        key_features = [
            'pause_rate', 'hesitation_rate', 'wpm', 'type_token_ratio',
            'substitution_score', 'word_finding_difficulty', 'anomaly_score'
        ]
        
        # 1. Feature correlation heatmap
        plt.figure(figsize=(10, 8))
        numeric_df = features_df.select_dtypes(include=['number'])
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        # Handle categorical data properly
        for col in numeric_df.columns:
            if isinstance(numeric_df[col].dtype, CategoricalDtype):
                numeric_df[col] = numeric_df[col].astype(float)
                
        corr_matrix = numeric_df.corr()
        corr_matrix = corr_matrix.fillna(0)
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        corr_path = os.path.join(batch_dir, "correlation_heatmap.png")
        plt.savefig(corr_path)
        plt.close()
        
        with open(corr_path, "rb") as image_file:
            visualizations['correlation_heatmap'] = base64.b64encode(image_file.read()).decode('utf-8')
            
        # 2. Feature distributions
        for feature in key_features:
            if feature in features_df.columns:
                if features_df[feature].isna().all():
                    continue
                    
                plt.figure(figsize=(8, 6))
                if isinstance(features_df[feature].dtype, CategoricalDtype):
                    sns.countplot(x=features_df[feature])
                else:
                    sns.histplot(features_df[feature].dropna(), kde=True)
                    
                plt.title(f'{feature.replace("_", " ").title()} Distribution')
                plt.xlabel(feature)
                plt.ylabel('Count')
                feature_path = os.path.join(batch_dir, f"{feature}_distribution.png")
                plt.savefig(feature_path)
                plt.close()
                
                with open(feature_path, "rb") as image_file:
                    visualizations[f'{feature}_distribution'] = base64.b64encode(image_file.read()).decode('utf-8')
                    
        # 3. Risk score distribution
        if 'risk_score' in features_df.columns and not features_df['risk_score'].isna().all():
            plt.figure(figsize=(8, 6))
            sns.histplot(features_df['risk_score'].dropna(), kde=True)
            plt.title('Risk Score Distribution')
            plt.xlabel('Risk Score')
            plt.ylabel('Count')
            risk_path = os.path.join(batch_dir, "risk_score_distribution.png")
            plt.savefig(risk_path)
            plt.close()
            
            with open(risk_path, "rb") as image_file:
                visualizations['risk_score_distribution'] = base64.b64encode(image_file.read()).decode('utf-8')
                
        return visualizations
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print(traceback.format_exc())
        return {}

def calculate_feature_importance(features_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate feature importance based on correlation with risk score."""
    try:
        if 'risk_score' not in features_df.columns or len(features_df) < 3:
            # Return default importance values
            return {
                'pause_rate': 0.8,
                'hesitation_rate': 0.7,
                'type_token_ratio': 0.6,
                'wpm': 0.5,
                'word_finding_difficulty': 0.7,
                'substitution_score': 0.6,
                'incomplete_sentences': 0.5
            }
            
        # Calculate correlation with risk score
        numeric_df = features_df.select_dtypes(include=['number'])
        correlations = numeric_df.corr()['risk_score'].abs().sort_values(ascending=False)
        
        # Normalize to 0-1 range
        max_corr = correlations.max()
        if max_corr > 0:
            normalized = correlations / max_corr
        else:
            normalized = correlations
            
        # Convert to dictionary and filter out risk_score itself
        importance_dict = normalized.to_dict()
        if 'risk_score' in importance_dict:
            del importance_dict['risk_score']
            
        # Take top 7 features
        top_features = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:7])
        return top_features
    except Exception as e:
        print(f"Error calculating feature importance: {e}")
        return {
            'pause_rate': 0.8,
            'hesitation_rate': 0.7,
            'type_token_ratio': 0.6,
            'wpm': 0.5
        }

def extract_insights(features_df: pd.DataFrame) -> List[str]:
    """Extract key insights from the analysis."""
    insights = []
    try:
        if features_df.empty:
            return ["Insufficient data for analysis"]
            
        # 1. Risk distribution insight
        if 'risk_score' in features_df.columns:
            high_risk_count = len(features_df[features_df['risk_score'] > 70])
            high_risk_pct = high_risk_count / len(features_df) * 100
            insights.append(
                f"{high_risk_count} samples ({high_risk_pct:.1f}%) show high risk patterns "
                f"(risk score > 70), suggesting potential cognitive concerns."
            )
            
        # 2. Anomaly insight
        if 'is_anomaly' in features_df.columns:
            anomaly_count = features_df['is_anomaly'].sum()
            if anomaly_count > 0:
                anomaly_pct = anomaly_count / len(features_df) * 100
                insights.append(
                    f"{anomaly_count} samples ({anomaly_pct:.1f}%) were identified as anomalies "
                    f"based on unsupervised machine learning analysis."
                )
                
        # 3. Most common linguistic issues
        linguistic_features = [
            ('hesitation_rate', 0.1, 'Hesitation markers'),
            ('word_finding_difficulty', 0.2, 'Word finding difficulties'),
            ('incomplete_sentences', 0.3, 'Incomplete sentences'),
            ('substitution_score', 0.2, 'Word substitutions')
        ]
        
        for feature, threshold, label in linguistic_features:
            if feature in features_df.columns:
                issue_count = len(features_df[features_df[feature] > threshold])
                if issue_count > 0:
                    issue_pct = issue_count / len(features_df) * 100
                    insights.append(
                        f"{issue_count} samples ({issue_pct:.1f}%) exhibit elevated {label.lower()}, "
                        f"which can be an indicator of cognitive challenges."
                    )
                    
        # 4. Speech rate insight
        if 'wpm' in features_df.columns:
            slow_speech_count = len(features_df[features_df['wpm'] < 120])
            if slow_speech_count > 0:
                slow_speech_pct = slow_speech_count / len(features_df) * 100
                insights.append(
                    f"{slow_speech_count} samples ({slow_speech_pct:.1f}%) show slower than typical "
                    f"speech rates (<120 WPM), which may indicate processing difficulties."
                )
                
        # 5. Pause pattern insight
        if 'pause_rate' in features_df.columns:
            high_pause_count = len(features_df[features_df['pause_rate'] > 0.3])
            if high_pause_count > 0:
                high_pause_pct = high_pause_count / len(features_df) * 100
                insights.append(
                    f"{high_pause_count} samples ({high_pause_pct:.1f}%) demonstrate elevated pause rates, "
                    f"potentially indicating difficulties with word retrieval or thought organization."
                )
                
        # 6. Vocabulary diversity insight
        if 'type_token_ratio' in features_df.columns:
            low_ttr_count = len(features_df[features_df['type_token_ratio'] < 0.4])
            if low_ttr_count > 0:
                low_ttr_pct = low_ttr_count / len(features_df) * 100
                insights.append(
                    f"{low_ttr_count} samples ({low_ttr_pct:.1f}%) show reduced vocabulary diversity, "
                    f"which can be associated with word finding difficulties in cognitive decline."
                )
                
        return insights
    except Exception as e:
        print(f"Error extracting insights: {e}")
        return ["Error analyzing data for insights"]

def calculate_risk_score(features: Dict[str, Any]) -> float:
    """Calculate cognitive decline risk score using ML-augmented rules."""
    try:
        # Check if we have a trained model
        model_path = "isolation_forest_model.joblib"
        if os.path.exists(model_path):
            try:
                # Use the ML model for anomaly detection component
                model = joblib.load(model_path)
                
                # Get expected feature count from model
                expected_feature_count = model.n_features_in_
                
                # Prepare feature names - this is critical to match training data structure
                all_possible_features = [
                    'pause_rate', 'pause_count', 'pause_duration', 'pitch_mean', 
                    'pitch_std', 'speech_rate_acoustic', 'spectral_centroid', 
                    'spectral_rolloff', 'jitter', 'shimmer', 'duration',
                    'hesitation_rate', 'type_token_ratio', 'wpm', 'word_count',
                    'sentence_count', 'avg_words_per_sentence', 'filler_word_ratio',
                    'coherence_score', 'sentence_completion', 'word_finding_difficulty', 
                    'substitution_score', 'incomplete_sentences', 'anomaly_score'
                ]
                
                # Create a feature vector with all possible features, using 0 for missing ones
                feature_vector = np.zeros((1, expected_feature_count))
                
                # If expected_feature_count doesn't match our list, use first n features
                feature_names = all_possible_features[:expected_feature_count]
                
                # Fill the feature vector with available values
                for i, feat in enumerate(feature_names):
                    if i < expected_feature_count:  # Safety check
                        feature_vector[0][i] = features.get(feat, 0)
                
                # Get anomaly score (-1 for anomalies, 1 for normal)
                anomaly_prediction = model.predict(feature_vector)[0]
                anomaly_score = 0 if anomaly_prediction == 1 else 1  # Convert to 0 (normal) or 1 (anomaly)
                
                # Base risk score starts at 40 with ML component adding up to 20 points
                base_score = 40 + (anomaly_score * 20)
            except Exception as e:
                print(f"Error using ML model for risk scoring: {e}")
                print(traceback.format_exc())  # Add detailed traceback
                base_score = 50  # Fallback to base score
        else:
            # If no model exists, start with base score of 50
            base_score = 50
            
        # Rest of the function remains the same
        # Define feature weights
        modifiers = {
            'pause_rate': 10,  # Higher pause rate → higher risk
            'hesitation_rate': 15,  # More hesitations → higher risk
            'wpm': -10,  # Lower WPM → higher risk
            'type_token_ratio': -15,  # Lower vocabulary diversity → higher risk
            'filler_word_ratio': 10,  # More filler words → higher risk
            'word_finding_difficulty': 15,  # Word finding difficulties → higher risk
            'substitution_score': 12,  # Word substitutions → higher risk
            'incomplete_sentences': 10  # Incomplete sentences → higher risk
        }
        
        # Apply modifiers
        for feature, weight in modifiers.items():
            if feature in features:
                # Normalize feature between 0-1 (simplistic approach)
                value = features[feature]
                normalized = min(1.0, max(0.0, value))
                
                if feature == 'pause_rate':
                    normalized = min(1.0, value / 0.3)  # Assuming 0.3 is a high pause rate
                elif feature == 'hesitation_rate':
                    normalized = min(1.0, value / 0.1)  # Assuming 10% hesitation is high
                elif feature == 'wpm':
                    # Slower speech increases risk (normalize against typical 150 WPM)
                    normalized = 1.0 - min(1.0, value / 150.0)
                elif feature == 'type_token_ratio':
                    # Lower TTR increases risk (invert the value)
                    normalized = 1.0 - normalized
                
                # Apply weight
                base_score += normalized * weight
                
        # Ensure score is between 0-100
        final_score = max(0, min(100, base_score))
        return final_score
    except Exception as e:
        print(f"Error calculating risk score: {e}")
        print(traceback.format_exc())  # Add detailed traceback
        return 50.0  # Return default middle score on error

@app.post("/analyze", response_class=JSONResponse)
async def analyze_speech(audio_file: UploadFile = File(...)):
    """
    Analyze speech audio file for cognitive decline markers.
    
    - **audio_file**: WAV file containing speech to be analyzed
    
    Returns:
    - risk_score: Overall risk score (0-100)
    - transcription: Transcribed text
    - key_indicators: Important speech metrics
    - ml_results: Results from unsupervised ML analysis
    """
    if not audio_file.filename.endswith(('.wav', '.WAV')):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    # Save the uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        shutil.copyfileobj(audio_file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Extract features
        audio_features = extract_audio_features(temp_path)
        
        # Get text and ensure it's not None
        text = speech_to_text(temp_path)
        if text is None:
            text = ""  # Ensure text is never None
            print(f"Warning: Speech-to-text returned None for {audio_file.filename}, using empty string")
        
        # Extract text features with validated text
        text_features = extract_linguistic_features(text)
        recall_features = detect_word_recall_issues(text)
        
        # Calculate WPM with validated inputs
        wpm = calculate_wpm(text, audio_features.get('duration', 1))
        text_features['wpm'] = wpm
        
        # Combine features
        combined_features = {**audio_features, **text_features, **recall_features}
        
        # Calculate risk score
        risk_score = calculate_risk_score(combined_features)
        combined_features['risk_score'] = risk_score
        
        # Create a mini dataframe for this sample for ML analysis
        sample_df = pd.DataFrame([combined_features])
        
        # Apply unsupervised ML (though with one sample, this is mostly for consistency)
        ml_results = apply_unsupervised_ml(sample_df)
        combined_features['anomaly_score'] = ml_results['anomaly_scores'][0]
        combined_features['is_anomaly'] = ml_results['is_anomaly'][0]
        
        # Store result in memory (no database)
        sample_id = str(uuid.uuid4())[:8]
        combined_features['sample_id'] = sample_id
        analysis_results[sample_id] = combined_features
        
        # Create response
        response = {
            'risk_score': risk_score,
            'transcription': text,
            'key_indicators': {
                'pause_rate': audio_features.get('pause_rate', 0),
                'hesitation_rate': text_features.get('hesitation_rate', 0),
                'wpm': wpm,
                'vocabulary_diversity': text_features.get('type_token_ratio', 0),
                'filler_word_ratio': text_features.get('filler_word_ratio', 0),
                'word_finding_difficulty': recall_features.get('word_finding_difficulty', 0),
                'substitution_score': recall_features.get('substitution_score', 0),
                'incomplete_sentences': recall_features.get('incomplete_sentences', 0)
            },
            'ml_results': {
                'anomaly_score': ml_results['anomaly_scores'][0],
                'is_anomaly': ml_results['is_anomaly'][0]
            },
            'sample_id': sample_id
        }
        
        return response
    except Exception as e:
        print(f"Error processing audio: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)



@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    """Batch analysis page."""
    return templates.TemplateResponse("batch.html", {"request": request})

@app.post("/analyze-batch", response_class=JSONResponse)
async def analyze_batch(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Analyze multiple speech audio files in batch mode.
    
    - **files**: List of WAV files containing speech to be analyzed
    
    Returns:
    - batch_id: ID for retrieving batch analysis results
    - sample_count: Number of samples in the batch
    - status: Processing status
    """
    # Check file types
    for file in files:
        if not file.filename.endswith(('.wav', '.WAV')):
            raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    # Generate batch ID
    batch_id = str(uuid.uuid4())
    
    # Create a background task for processing
    background_tasks.add_task(process_batch, files, batch_id)
    
    return {
        "batch_id": batch_id,
        "sample_count": len(files),
        "status": "processing",
        "message": "Batch processing started. Use /batch-results/{batch_id} to check status and get results."
    }

async def process_batch(files: List[UploadFile], batch_id: str):
    """Process a batch of audio files in the background (in-memory only)."""
    results = []
    failed_files = []
    
    for file in files:
        try:
            # Save file temporarily
            with NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name
            
            # Process file
            audio_features = extract_audio_features(temp_path)
            text = speech_to_text(temp_path) or ""  # Ensure text is not None
            text_features = extract_linguistic_features(text)
            recall_features = detect_word_recall_issues(text)
            
            # Calculate WPM
            wpm = calculate_wpm(text, audio_features.get('duration', 1))
            text_features['wpm'] = wpm
            
            # Combine features and convert numpy types
            combined_features = {
                **audio_features,
                **text_features, 
                **recall_features,
                'risk_score': calculate_risk_score({**audio_features, **text_features, **recall_features}),
                'sample_id': file.filename,
                'processed_at': datetime.now().isoformat()
            }
            
            # Convert numpy types
            for k, v in combined_features.items():
                if isinstance(v, (np.integer, np.int32, np.int64)):
                    combined_features[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    combined_features[k] = float(v)
                elif isinstance(v, np.ndarray):
                    combined_features[k] = v.tolist()
            
            results.append(combined_features)
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            failed_files.append({
                'filename': file.filename,
                'error': str(e)
            })
            continue
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # Generate visualizations in memory
    if results:
        df = pd.DataFrame(results)
        visualizations = generate_visualizations(df, batch_id)
        
        # Generate report HTML
        report_html = generate_report(df, visualizations, batch_id)
        
        # Store results in memory
        batch_analysis_results[batch_id] = {
            "status": "completed",
            "sample_count": len(results),
            "failed_count": len(failed_files),
            "created_at": datetime.now().isoformat(),
            "results": results,
            "failed_files": failed_files,
            "visualizations": visualizations,
            "report_html": report_html,
            "has_data": bool(results)
        }
    else:
        batch_analysis_results[batch_id] = {
            "status": "completed",
            "sample_count": 0,
            "failed_count": len(failed_files),
            "created_at": datetime.now().isoformat(),
            "results": [],
            "failed_files": failed_files,
            "visualizations": {},
            "report_html": "<html><body><h1>No Results</h1><p>No files were successfully processed</p></body></html>",
            "has_data": False
        }

@app.get("/batch-results/{batch_id}")
async def get_batch_results(batch_id: str):
    try:
        batch_info = batch_analysis_results.get(batch_id)
        
        if not batch_info:
            raise HTTPException(status_code=404, detail="Batch job not found")
        
        # Return minimal information
        return {
            "batch_id": batch_id,
            "status": batch_info.get("status", "unknown"),
            "sample_count": batch_info.get("sample_count", 0),
            "created_at": batch_info.get("created_at"),
            "has_data": batch_info.get("has_data", False)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving batch results: {str(e)}"
        )

@app.get("/batch-report/{batch_id}", response_class=HTMLResponse)
async def get_batch_report(batch_id: str):
    """
    Get HTML report from a batch analysis job.
    
    - **batch_id**: ID of the batch processing job
    
    Returns:
    - HTML report with visualizations and insights
    """
    try:
        batch_info = batch_analysis_results.get(batch_id)

        if not batch_info:
            return HTMLResponse(content="<html><body><h1>Error</h1><p>Batch job not found</p></body></html>", status_code=404)

        if batch_info.get("status") != "completed":
            return HTMLResponse(
                content=f"""
                <html>
                    <body>
                        <h1>Batch Processing in Progress</h1>
                        <p>Status: {batch_info.get("status", "unknown")}</p>
                        <p>Please check back later.</p>
                    </body>
                </html>
                """,
                status_code=202
            )

        if not batch_info.get("report_html"):
            return HTMLResponse(
                content="<html><body><h1>Error</h1><p>Report not available</p></body></html>",
                status_code=500
            )

        return HTMLResponse(content=batch_info["report_html"])
    except Exception as e:
        print(f"Error retrieving batch report: {e}")
        return HTMLResponse(
            content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>",
            status_code=500
        )

def generate_report(features_df: pd.DataFrame, visualizations: Dict[str, str], batch_id: str) -> str:
    """Generate HTML report for batch analysis."""
    try:
        # Calculate summary statistics
         # Calculate only essential statistics for large datasets
        if len(features_df) > 50:
            features_df = features_df.sample(50)
        summary = {
            "sample_count": len(features_df),
            "high_risk_count": len(features_df[features_df['risk_score'] >= 70]) if 'risk_score' in features_df else 0,
            "anomaly_count": features_df['is_anomaly'].sum() if 'is_anomaly' in features_df else 0,
            "avg_risk": features_df['risk_score'].mean() if 'risk_score' in features_df else 0
        }
        
        # Calculate feature averages
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        feature_avgs = {col: features_df[col].mean() for col in numeric_cols}
        
        # Feature descriptions
        feature_descriptions = {
            'pause_rate': 'Frequency of pauses in speech (higher values may indicate cognitive load)',
            'hesitation_rate': 'Frequency of hesitation markers like "um" and "uh"',
            'wpm': 'Words per minute (speech rate)',
            'type_token_ratio': 'Vocabulary diversity (higher values indicate more diverse word usage)',
            'word_finding_difficulty': 'Measure of word retrieval difficulties',
            'substitution_score': 'Measure of inappropriate word substitutions',
            'incomplete_sentences': 'Frequency of incomplete or fragmented sentences'
        }
        
        # Feature importance
        feature_importance = calculate_feature_importance(features_df)
        feature_names = list(feature_importance.keys())
        feature_importances = list(feature_importance.values())
        
        # Insights
        insights = extract_insights(features_df)
        
        # Render template
        return templates.get_template("report.html").render({
            "request": None,  # No request object available in background task
            "batch_id": batch_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "feature_avgs": feature_avgs,
            "feature_descriptions": feature_descriptions,
            "feature_names": feature_names,
            "feature_importances": feature_importances,
            "visualizations": visualizations,
            "insights": insights
        })
    except Exception as e:
        print(f"Error generating report: {e}")
        return """
        <html>
            <body>
                <h1>Error Generating Report</h1>
                <p>An error occurred while generating the report.</p>
            </body>
        </html>
        """


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Home page with single file analysis form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8080)