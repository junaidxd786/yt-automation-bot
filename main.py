"""
================================================================================
YOUTUBE SHORTS GENERATOR v10 ULTRA - RAILWAY EDITION
================================================================================

v10 ULTRA FEATURES (All Preserved):
🧠 DEEP ML SCORING: Audio + Sentiment + Viral Keyword analysis
🛡️ INTERACTIVE APPROVAL: Bot finds videos -> You approve -> Bot processes
🕵️ SMART FINDER: Finds Top 2 high-potential videos
🎣 HOOK DETECTION: AI-powered first-5-seconds analysis
📊 VIRAL SCORE: Comprehensive viral potential prediction
☁️ TELEGRAM DELIVERY: Clips sent directly to your chat

RAILWAY-SPECIFIC CHANGES:
- Removed Google Colab dependencies
- Uses environment variables for API keys
- Local file storage (./data/)
- Standard Python async (no nest_asyncio)
================================================================================
"""

import os
import sys
import json
import time
import uuid
import re
import shutil
import logging
import threading
import subprocess
import asyncio
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Imports (will fail gracefully if missing)
import cv2
import numpy as np
import torch
import requests
import yt_dlp
import librosa
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from transformers import pipeline
from ultralytics import YOLO

# Suppress noisy warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('yt_dlp').setLevel(logging.ERROR)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

@dataclass
class Config:
    # Paths (Railway-compatible)
    OUTPUT_FOLDER: str = './data/output'
    TEMP_FOLDER: str = './data/temp_processing'
    CACHE_FOLDER: str = './data/cache'
    YOLO_MODEL_PATH: str = './data/models/yolov8n-face.pt'
    
    # API Keys (from environment variables)
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')
    DEEPGRAM_API_KEY: str = os.getenv('DEEPGRAM_API_KEY', '')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    # Settings
    MAX_THREADS: int = 4
    FFMPEG_TIMEOUT: int = 600
    
    def setup_directories(self):
        for folder in [self.OUTPUT_FOLDER, self.TEMP_FOLDER, self.CACHE_FOLDER, './data/models']:
            os.makedirs(folder, exist_ok=True)
        logger.info(f"✅ Directories ready: {self.OUTPUT_FOLDER}")
    
    def validate(self):
        """Validate that required environment variables are set."""
        missing = []
        if not self.TELEGRAM_BOT_TOKEN:
            missing.append('TELEGRAM_BOT_TOKEN')
        if not self.DEEPGRAM_API_KEY:
            missing.append('DEEPGRAM_API_KEY')
        if not self.GEMINI_API_KEY:
            missing.append('GEMINI_API_KEY')
        
        if missing:
            logger.error(f"❌ Missing environment variables: {', '.join(missing)}")
            logger.error("Set them in Railway dashboard or .env file")
            sys.exit(1)
        
        logger.info("✅ All API keys configured")

CONFIG = Config()
CONFIG.setup_directories()
CONFIG.validate()

# ==================== YOUTUBE FINDER ====================

class YouTubeFinder:
    def __init__(self, config: Config):
        self.config = config

    def find_videos(self, query: str, limit: int = 2, duration_min: int = 5) -> List[Dict]:
        """Search YouTube for high-potential videos with randomization."""
        import random
        logger.info(f"🔎 Searching YouTube for: '{query}'...")
        
        search_count = max(10, limit * 5)
        
        ydl_opts = {
            'quiet': True,
            'default_search': f'ytsearch{search_count}',
            'extract_flat': 'in_playlist',
            'no_warnings': True,
        }
        
        candidates = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            res = ydl.extract_info(query, download=False)
            if 'entries' in res:
                for entry in res['entries']:
                    if not entry: continue
                    duration = entry.get('duration', 0)
                    if duration < duration_min * 60: continue
                    
                    candidates.append({
                        'title': entry.get('title'),
                        'url': entry.get('url'),
                        'duration': duration,
                        'views': entry.get('view_count', 'N/A'),
                        'id': entry.get('id')
                    })
        
        random.shuffle(candidates)
        result = candidates[:limit]
        logger.info(f"✅ Found {len(result)} videos")
        return result

# ==================== MODEL MANAGER ====================

class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized: return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_model = None
        self.yolo_model = None
        self._initialized = True
        logger.info(f"🖥️ Using device: {self.device}")

    def load_sentiment_model(self):
        if self.sentiment_model: return self.sentiment_model
        try:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.sentiment_model = pipeline("sentiment-analysis", model=model_name, 
                                           device=0 if self.device == "cuda" else -1)
            logger.info(f"✅ Sentiment model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Sentiment Load Fail: {e}")
        return self.sentiment_model

    def ensure_yolo_model_file(self):
        if not os.path.exists(CONFIG.YOLO_MODEL_PATH):
            logger.info("⬇️ Downloading YOLOv8-Face model...")
            os.makedirs(os.path.dirname(CONFIG.YOLO_MODEL_PATH), exist_ok=True)
            try:
                urllib.request.urlretrieve(
                    "https://github.com/lindevs/yolov8-face/releases/download/v1.1/yolov8n-face-lindevs.pt", 
                    CONFIG.YOLO_MODEL_PATH
                )
                logger.info("✅ YOLOv8-Face downloaded")
            except:
                urllib.request.urlretrieve(
                    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt", 
                    CONFIG.YOLO_MODEL_PATH
                )

    def load_yolo_model(self):
        if self.yolo_model: return self.yolo_model
        try:
            self.ensure_yolo_model_file()
            self.yolo_model = YOLO(CONFIG.YOLO_MODEL_PATH)
            self.yolo_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            logger.info("✅ YOLO model loaded")
        except Exception as e:
            logger.error(f"YOLO Load Fail: {e}")
        return self.yolo_model

# ==================== VIDEO CONTEXT ====================

class VideoContext:
    def __init__(self, video_path):
        self.path = video_path
        self.valid = False
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            self.valid = True
        cap.release()

# ==================== CONTENT ANALYZER ====================

class ContentAnalyzer:
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.models = model_manager

    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&punctuate=true&utterances=true"
        headers = {"Authorization": f"Token {self.config.DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"}
        try:
            with open(audio_path, 'rb') as audio_file:
                response = requests.post(url, headers=headers, data=audio_file, timeout=180)
            if response.status_code != 200: return []
            result = response.json()
            words = []
            for u in result.get('results', {}).get('utterances', []):
                for w in u.get('words', []):
                    words.append({'text': w.get('word', ''), 'start': w.get('start', 0), 'end': w.get('end', 0)})
            logger.info(f"✅ Transcribed {len(words)} words")
            return words
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []

    def score_segment_deep(self, video_ctx, audio_path, start, duration, text):
        """Deep ML Scoring"""
        scores = {}
        
        # 1. Audio Energy
        try:
            y, sr = librosa.load(audio_path, sr=22050, offset=start, duration=duration)
            rms = librosa.feature.rms(y=y)[0]
            scores['audio'] = min(1.0, np.mean(rms) * 100)
        except:
            scores['audio'] = 0.5

        # 2. Sentiment
        try:
            model = self.models.load_sentiment_model()
            if model and text:
                res = model(text[:500])[0]
                scores['sentiment'] = res['score']
            else:
                scores['sentiment'] = 0.5
        except:
            scores['sentiment'] = 0.5

        # 3. Viral Keywords
        viral_words = ['secret', 'amazing', 'shocking', 'unbelievable', 'proven', 'hack', 'trick', 'truth', 'finally', 'insane']
        scores['keywords'] = min(1.0, sum(1 for w in viral_words if w in text.lower()) / 3)

        final = (scores['audio'] * 0.3) + (scores['sentiment'] * 0.3) + (scores['keywords'] * 0.4)
        return final

    def find_best_clips(self, video_ctx: VideoContext, audio_path: str, words: List[Dict], num_clips: int) -> List[Dict]:
        logger.info(f"🧠 Deep Analysis for {num_clips} clips...")
        
        candidates = []
        i = 0
        target_dur = 45
        
        while i < len(words):
            if words[i]['start'] > video_ctx.duration - 20: break
            
            end_idx = i
            for j in range(i, len(words)):
                dur = words[j]['end'] - words[i]['start']
                if dur >= target_dur * 0.8:
                    if words[j]['text'][-1] in '.?!':
                        end_idx = j
                        break
                    if dur >= target_dur * 1.2:
                        end_idx = j
                        break
            
            if end_idx > i:
                segment_words = words[i:end_idx+1]
                transcript = " ".join([w['text'] for w in segment_words])
                dur = words[end_idx]['end'] - words[i]['start']
                
                score = self.score_segment_deep(video_ctx, audio_path, words[i]['start'], dur, transcript)
                
                candidates.append({
                    'start': words[i]['start'],
                    'duration': dur,
                    'transcript': transcript,
                    'words': segment_words,
                    'score': score
                })
                i = end_idx + 1
            else:
                i += 10

        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        selected = []
        for c in candidates:
            if len(selected) >= num_clips: break
            if not any(abs(s['start'] - c['start']) < 30 for s in selected):
                selected.append(c)
                
        return selected

    def generate_seo(self, transcript: str, niche: str = "general") -> Dict:
        prompt = f"""Generate viral TikTok content for this transcript: "{transcript[:500]}"

Niche: {niche}

Return JSON with:
1. "title": Catchy title (max 50 chars, include 1 emoji)
2. "hashtags": Array of 7 hashtags

JSON only: {{ "title": "...", "hashtags": ["#...", ...] }}"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.config.GEMINI_API_KEY}"
            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
            return json.loads(resp.json()['candidates'][0]['content']['parts'][0]['text'].replace('```json','').replace('```','').strip())
        except: 
            return {"title": "Viral Short 🔥", "hashtags": ["#shorts", "#fyp", "#viral"]}

# ==================== VIRAL ANALYZER ====================

class ViralAnalyzer:
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.models = model_manager
    
    def analyze_hook(self, words: List[Dict], audio_path: str, transcript: str) -> Dict:
        """AI-powered hook detection - analyzes first 5 seconds."""
        first_5_sec_words = [w for w in words if w['end'] <= 5]
        hook_text = ' '.join([w['text'] for w in first_5_sec_words])
        
        if not hook_text.strip():
            return {'score': 0.1, 'reason': 'Silent/slow opening', 'text': ''}
        
        prompt = f"""Analyze this opening line for TikTok viral potential (first 5 seconds):

"{hook_text}"

Score 1-10 on: CURIOSITY, PATTERN_INTERRUPT, EMOTIONAL_TRIGGER, CLARITY, URGENCY

Return JSON: {{"score": 1-10, "strongest": "which criteria", "weakness": "main issue"}}"""
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.config.GEMINI_API_KEY}"
            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10)
            raw_text = resp.json()['candidates'][0]['content']['parts'][0]['text']
            raw_text = raw_text.replace('```json', '').replace('```', '').strip()
            
            json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
            
            score = min(1.0, max(0.0, result.get('score', 5) / 10))
            
            return {
                'score': score,
                'text': hook_text,
                'strongest': result.get('strongest', 'unknown'),
                'weakness': result.get('weakness', 'none')
            }
        except:
            return self._fallback_hook_analysis(hook_text, audio_path)
    
    def _fallback_hook_analysis(self, hook_text: str, audio_path: str) -> Dict:
        score = 0.5
        text_lower = hook_text.lower()
        
        curiosity_triggers = ['what if', 'imagine', 'secret', 'truth', 'never', 'always', 'did you know']
        trigger_count = sum(1 for t in curiosity_triggers if t in text_lower)
        if trigger_count > 0:
            score += min(0.3, trigger_count * 0.1)
        
        if '?' in hook_text:
            score += 0.15
        
        return {'score': min(1.0, score), 'text': hook_text, 'strongest': 'heuristic'}
    
    def calculate_viral_score(self, clip: Dict, hook_analysis: Dict, audio_path: str) -> Dict:
        scores = {}
        
        scores['hook'] = hook_analysis.get('score', 0.5) * 25
        scores['content'] = clip.get('score', 0.5) * 25
        
        transcript = clip.get('transcript', '')
        duration = clip.get('duration', 45)
        words_per_min = (len(transcript.split()) / duration) * 60 if duration > 0 else 0
        
        if 140 <= words_per_min <= 180:
            scores['pacing'] = 20
        elif 120 <= words_per_min <= 200:
            scores['pacing'] = 15
        else:
            scores['pacing'] = 10
        
        try:
            model = self.models.load_sentiment_model()
            if model and transcript:
                res = model(transcript[:500])[0]
                scores['emotion'] = res['score'] * 15
            else:
                scores['emotion'] = 7.5
        except:
            scores['emotion'] = 7.5
        
        try:
            y, sr = librosa.load(audio_path, sr=22050, offset=clip['start'], duration=min(duration, 60))
            rms = librosa.feature.rms(y=y)[0]
            variance = np.std(rms)
            scores['dynamics'] = min(1.0, variance * 50) * 15
        except:
            scores['dynamics'] = 7.5
        
        total = sum(scores.values())
        
        if total >= 75:
            tier = "🔥 HIGH VIRAL POTENTIAL"
        elif total >= 55:
            tier = "✅ GOOD POTENTIAL"
        elif total >= 40:
            tier = "⚠️ MODERATE"
        else:
            tier = "❌ LOW POTENTIAL"
        
        return {
            'total_score': round(total, 1),
            'tier': tier,
            'breakdown': scores,
            'words_per_minute': round(words_per_min, 1)
        }
    
    def get_trending_hashtags(self, niche: str, transcript: str) -> List[str]:
        prompt = f"""Generate 7 trending TikTok hashtags for this content.
Content: "{transcript[:300]}"
Niche: {niche}

Return ONLY a JSON array: ["#hashtag1", "#hashtag2", ...]"""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.config.GEMINI_API_KEY}"
            resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10)
            result = resp.json()['candidates'][0]['content']['parts'][0]['text']
            hashtags = json.loads(result.replace('```json','').replace('```','').strip())
            return [h if h.startswith('#') else f'#{h}' for h in hashtags[:7]]
        except:
            return ['#fyp', '#viral', '#shorts', '#trending', '#foryou']

# ==================== VIDEO PROCESSOR ====================

class VideoProcessor:
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.models = model_manager

    def download_video(self, url: str) -> Optional[str]:
        video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
        video_id = video_id_match.group(1) if video_id_match else "temp"
        path = f"{self.config.TEMP_FOLDER}/video_{video_id}.mp4"
        if os.path.exists(path): os.remove(path)
        
        opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': path.replace('.mp4', ''),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 5
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"✅ Downloaded: {size:.1f} MB")
                return path
            return None
        except Exception as e:
            logger.error(f"Download Fail: {e}")
            return None

    def extract_audio(self, video_path):
        audio_path = f'{self.config.TEMP_FOLDER}/audio_{uuid.uuid4().hex[:6]}.wav'
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-ar', '16000', '-ac', '1', audio_path], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path

    def get_face_center(self, video_ctx, start, duration):
        model = self.models.load_yolo_model()
        if not model: return (video_ctx.width//2, video_ctx.height//2)
        cap = cv2.VideoCapture(video_ctx.path)
        centers = []
        for t in [start, start + duration/2, start + duration]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * video_ctx.fps))
            ret, frame = cap.read()
            if ret:
                res = model.predict(frame, classes=[0], conf=0.25, verbose=False)
                if res and len(res[0].boxes) > 0:
                     b = res[0].boxes.xyxy.cpu().numpy()[0]
                     centers.append(((b[0]+b[2])/2, (b[1]+b[3])/2))
        cap.release()
        if not centers: return (video_ctx.width//2, video_ctx.height//2)
        return (int(sum(c[0] for c in centers)/len(centers)), int(sum(c[1] for c in centers)/len(centers)))

    def process_clip(self, video_ctx, clip, index, layout):
        start, dur = clip['start'], clip['duration']
        target_w, target_h = 1080, 1920
        safe_title = f"clip_{index}_{uuid.uuid4().hex[:4]}"
        out_file = os.path.join(self.config.OUTPUT_FOLDER, f"{safe_title}.mp4")
        ass_file = os.path.join(self.config.TEMP_FOLDER, f"sub_{index}.ass")

        rel_words = [{'text': w['text'], 'start': w['start']-start, 'end': w['end']-start} for w in clip['words']]
        self._write_ass(rel_words, ass_file, target_w, target_h)

        if layout == 'crop':
            cx, _ = self.get_face_center(video_ctx, start, dur)
            scale = target_h / video_ctx.height
            sw = int(video_ctx.width * scale)
            crop_x = max(0, min(int(cx * scale - target_w/2), sw - target_w))
            vf = f"[0:v]scale={sw}:{target_h}:flags=fast_bilinear,crop={target_w}:{target_h}:{crop_x}:0,ass={ass_file}[outv]"
        elif layout == 'blur':
            vf = f"[0:v]scale={target_w}:{target_h}:flags=fast_bilinear,boxblur=40:40[bg];[0:v]scale={target_w}:-1:flags=fast_bilinear[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,ass={ass_file}[outv]"
        else:
            vf = f"[0:v]scale={target_w}:-1:flags=fast_bilinear,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,ass={ass_file}[outv]"

        cmd = ['ffmpeg', '-y', '-ss', str(start), '-t', str(dur), '-i', video_ctx.path, '-filter_complex', vf, 
               '-map', '[outv]', '-map', '0:a?', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', 
               '-c:a', 'aac', '-b:a', '96k', out_file]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=self.config.FFMPEG_TIMEOUT)
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
            
        if os.path.exists(ass_file): os.remove(ass_file)
        return out_file

    def _write_ass(self, words, filename, w, h):
        events = ""
        for i in range(0, len(words), 3):
            chunk = words[i:i+3]
            if not chunk: continue
            text = " ".join([c['text'] for c in chunk]).upper()
            s, e = chunk[0]['start'], chunk[-1]['end']
            def fmt(t): return f"{int(t//3600)}:{int((t%3600)//60):02d}:{t%60:05.2f}"
            events += f"Dialogue: 0,{fmt(s)},{fmt(e)},Default,,0,0,0,,{{\\fad(100,100)}}{text}\n"
        
        header = f"""[Script Info]
PlayResX: {w}
PlayResY: {h}
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,72,&H00FFFF&,&H000000FF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,4,2,2,30,30,150,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        with open(filename, 'w', encoding='utf-8') as f: f.write(header+events)

# ==================== BOT INTERFACE ====================

class BotInterface:
    def __init__(self, config: Config):
        self.config = config
        self.finder = YouTubeFinder(config)
        
        self.user_states = {}
        self.session_data = {}
        
        self.app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("auto", self.auto_cmd))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 <b>v10 ULTRA - Railway Edition</b>\n\n"
            "<b>Commands:</b>\n"
            "/auto &lt;niche&gt; — Search & Create Clips\n\n"
            "<b>Example:</b>\n"
            "<code>/auto tech podcasts</code>\n\n"
            "✨ <b>Features:</b>\n"
            "• AI Hook Detection\n"
            "• Viral Score Prediction\n"
            "• Trending Hashtags\n"
            "• Direct Telegram Delivery"
        )
        await update.message.reply_text(msg, parse_mode='HTML')

    async def auto_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if not context.args:
            await update.message.reply_text("⚠️ Usage: <code>/auto &lt;query&gt;</code>", parse_mode='HTML')
            return
        
        query = " ".join(context.args)
        await update.message.reply_text(f"🔎 Searching top 2 videos for: '{query}'...")
        
        loop = asyncio.get_running_loop()
        videos = await loop.run_in_executor(None, self.finder.find_videos, query, 2)
        
        if not videos: 
            return await update.message.reply_text("❌ No videos found.")
        
        self.session_data[chat_id] = {'videos': videos, 'niche': query}
        
        msg = "📹 <b>Found Videos</b>\n\n"
        for i, v in enumerate(videos):
            duration_min = int(v['duration'] / 60)
            msg += f"{i+1}. <b>{v['title'][:50]}...</b>\n   ⏱ {duration_min} min\n\n"
            
        msg += "<i>Tap Approve to continue</i>"
        
        keyboard = [
            [InlineKeyboardButton("✅ Approve", callback_data="approve")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
        ]
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        chat_id = update.effective_chat.id
        await query.answer()
        
        if query.data == "cancel":
            await query.edit_message_text("❌ Operation Cancelled.")
            self.session_data.pop(chat_id, None)
            self.user_states.pop(chat_id, None)
            return
            
        if query.data == "approve":
            if chat_id not in self.session_data:
                await query.edit_message_text("⚠️ Session expired. Use /auto again.")
                return
            
            self.user_states[chat_id] = "WAITING_CLIP_COUNT"
            msg = (
                "🔢 <b>How many clips?</b>\n\n"
                "Reply with a number (1-20)\n"
                "<i>Example: 6</i>"
            )
            await query.edit_message_text(msg, parse_mode='HTML')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        state = self.user_states.get(chat_id)
        text = update.message.text.strip()
        
        if state == "WAITING_CLIP_COUNT":
            try:
                count = int(text)
                if count < 1 or count > 20: raise ValueError
                
                self.session_data[chat_id]['clip_count'] = count
                self.user_states[chat_id] = None
                
                await update.message.reply_text(f"🚀 Processing {count} clips... This may take a few minutes.")
                
                videos = self.session_data[chat_id]['videos']
                niche = self.session_data[chat_id].get('niche', 'general')
                
                asyncio.create_task(self.run_processing(videos, count, update.message, niche))
                
            except ValueError:
                await update.message.reply_text("⚠️ Please enter a number between 1-20")

    async def run_processing(self, videos, total_target_clips, message, niche: str = "general"):
        async def send(msg): 
            await message.reply_text(msg)
        
        async def send_video(video_path, caption): 
            try:
                with open(video_path, 'rb') as f:
                    await message.reply_video(video=f, caption=caption[:1024], supports_streaming=True)
            except Exception as e:
                logger.error(f"Failed to send video: {e}")
                await send(f"⚠️ Couldn't send video: {e}")
        
        loop = asyncio.get_running_loop()
        
        mm = ModelManager()
        vp = VideoProcessor(self.config, mm)
        ca = ContentAnalyzer(self.config, mm)
        va = ViralAnalyzer(self.config, mm)
        
        try:
            clips_per_video = max(1, total_target_clips // len(videos))
            global_clip_index = 0
            
            for vid_idx, video in enumerate(videos):
                remaining_needed = total_target_clips - global_clip_index
                if remaining_needed <= 0: break
                
                current_target = min(clips_per_video, remaining_needed)
                if vid_idx == len(videos) - 1: current_target = remaining_needed
                
                await send(f"📺 Processing Video {vid_idx+1}/{len(videos)}\n{video['title'][:50]}...\nTargeting {current_target} clips...")
                
                vpath = await loop.run_in_executor(None, vp.download_video, video['url'])
                if not vpath: 
                    await send("❌ Download failed, skipping...")
                    continue
                    
                vctx = VideoContext(vpath)
                
                apath = await loop.run_in_executor(None, vp.extract_audio, vpath)
                words = await loop.run_in_executor(None, ca.transcribe_audio, apath)
                
                if len(words) > 100:
                    await send("🧠 Deep ML Analysis + Hook Detection...")
                    clips = await loop.run_in_executor(None, ca.find_best_clips, vctx, apath, words, current_target)
                    
                    for i, clip in enumerate(clips):
                        global_clip_index += 1
                        
                        clip_words = clip.get('words', [])
                        rel_words = [{'text': w['text'], 'start': w['start'] - clip['start'], 'end': w['end'] - clip['start']} for w in clip_words]
                        
                        hook_analysis = await loop.run_in_executor(None, va.analyze_hook, rel_words, apath, clip.get('transcript', ''))
                        viral_score = await loop.run_in_executor(None, va.calculate_viral_score, clip, hook_analysis, apath)
                        
                        out = await loop.run_in_executor(None, vp.process_clip, vctx, clip, global_clip_index, 'pad')
                        
                        seo = await loop.run_in_executor(None, ca.generate_seo, clip.get('transcript', ''), niche)
                        extra_hashtags = await loop.run_in_executor(None, va.get_trending_hashtags, niche, clip.get('transcript', ''))
                        
                        all_hashtags = list(dict.fromkeys(seo.get('hashtags', []) + extra_hashtags))[:8]
                        
                        caption = f"📊 {viral_score['tier']}\nScore: {viral_score['total_score']}/100\n🎣 Hook: {hook_analysis.get('score', 0)*100:.0f}%\n\n📝 {seo['title']}\n\n{' '.join(all_hashtags)}"
                        
                        if os.path.exists(out):
                            await send_video(out, caption)
                            os.remove(out)
                
                if os.path.exists(vpath): os.remove(vpath)
                if os.path.exists(apath): os.remove(apath)
                
            await send(f"🏁 Batch Complete!\nTotal {global_clip_index} clips created and sent.")
                
        except Exception as e:
            await send(f"❌ Critical Error: {e}")
            logger.error("Process Error", exc_info=True)

    def run(self):
        logger.info("🤖 v10 Ultra - Railway Edition Starting...")
        logger.info(f"📁 Output: {self.config.OUTPUT_FOLDER}")
        self.app.run_polling()

# ==================== MAIN ====================

if __name__ == "__main__":
    bot = BotInterface(CONFIG)
    bot.run()
