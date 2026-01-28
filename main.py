"""
================================================================================
YOUTUBE SHORTS GENERATOR v10 ULTRA - RAILWAY EDITION (LIGHTWEIGHT)
================================================================================

FEATURES:
🧠 DEEP SCORING: Audio + Gemini Sentiment + Viral Keywords
🛡️ INTERACTIVE APPROVAL: Bot finds videos -> You approve -> Bot processes
🕵️ SMART FINDER: Finds Top 2 high-potential videos
🎣 HOOK DETECTION: AI-powered first-5-seconds analysis
📊 VIRAL SCORE: Comprehensive viral potential prediction
☁️ TELEGRAM DELIVERY: Clips sent directly to your chat

LIGHTWEIGHT VERSION:
- No PyTorch/YOLO/Transformers (saves 5GB)
- Uses Gemini API for all AI tasks
- Uses letterbox (pad) layout only
================================================================================
"""

import os
import sys
import json
import uuid
import re
import logging
import subprocess
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import requests
import yt_dlp
import librosa
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('yt_dlp').setLevel(logging.ERROR)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

@dataclass
class Config:
    OUTPUT_FOLDER: str = './data/output'
    TEMP_FOLDER: str = './data/temp_processing'
    
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '').strip()
    DEEPGRAM_API_KEY: str = os.getenv('DEEPGRAM_API_KEY', '').strip()
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '').strip()
    YOUTUBE_COOKIES: str = os.getenv('YOUTUBE_COOKIES', '').strip()
    
    FFMPEG_TIMEOUT: int = 600
    COOKIE_FILE: str = './data/cookies.txt'

    def setup_cookies(self):
        if self.YOUTUBE_COOKIES:
            try:
                # 1. Decode Base64 if needed
                content = self.YOUTUBE_COOKIES
                if "Netscape" not in content and "{" not in content and len(content) > 20:
                    try:
                        import base64
                        content = base64.b64decode(content).decode('utf-8')
                    except:
                        pass
                
                # 2. Convert JSON to Netscape (EditThisCookie format -> yt-dlp format)
                if content.strip().startswith('['):
                    try:
                        import json
                        cookies = json.loads(content)
                        lines = ["# Netscape HTTP Cookie File"]
                        for c in cookies:
                            domain = c.get('domain', '')
                            flag = 'TRUE' if domain.startswith('.') else 'FALSE'
                            path = c.get('path', '/')
                            secure = 'TRUE' if c.get('secure') else 'FALSE'
                            expiry = str(int(c.get('expirationDate', 0)))
                            name = c.get('name', '')
                            value = c.get('value', '')
                            lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}")
                        content = "\n".join(lines)
                        logger.info("✅ Converted JSON cookies to Netscape format")
                    except Exception as e:
                        logger.error(f"Cookie conversion error: {e}")

                with open(self.COOKIE_FILE, 'w') as f:
                    f.write(content)
                logger.info("✅ YouTube Cookies loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load cookies: {e}")

    def setup_directories(self):
        for folder in [self.OUTPUT_FOLDER, self.TEMP_FOLDER]:
            os.makedirs(folder, exist_ok=True)
        logger.info(f"✅ Directories ready")
    
    def validate(self):
        missing = []
        if not self.TELEGRAM_BOT_TOKEN: missing.append('TELEGRAM_BOT_TOKEN')
        if not self.DEEPGRAM_API_KEY: missing.append('DEEPGRAM_API_KEY')
        if not self.GEMINI_API_KEY: missing.append('GEMINI_API_KEY')
        
        if missing:
            logger.error(f"❌ Missing: {', '.join(missing)}")
            sys.exit(1)
        logger.info("✅ All API keys configured")

CONFIG = Config()
CONFIG.setup_directories()
CONFIG.setup_cookies()
CONFIG.validate()

# ==================== YOUTUBE FINDER ====================

class YouTubeFinder:
    def __init__(self, config: Config):
        self.config = config

    def find_videos(self, query: str, limit: int = 2, duration_min: int = 5) -> List[Dict]:
        import random
        logger.info(f"🔎 Searching: '{query}'...")
        
        ydl_opts = {
            'quiet': True,
            'default_search': f'ytsearch{max(10, limit * 5)}',
            'extract_flat': 'in_playlist',
            'no_warnings': True,
        }
        
        if os.path.exists(self.config.COOKIE_FILE):
             ydl_opts['cookiefile'] = self.config.COOKIE_FILE
        
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
                        'id': entry.get('id')
                    })
        
        random.shuffle(candidates)
        result = candidates[:limit]
        logger.info(f"✅ Found {len(result)} videos")
        return result

# ==================== GEMINI HELPER ====================

def call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini API and return text response."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
        return resp.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return ""

# ==================== CONTENT ANALYZER ====================

class ContentAnalyzer:
    def __init__(self, config: Config):
        self.config = config

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

    def score_segment(self, audio_path: str, start: float, duration: float, text: str) -> float:
        """Score segment using audio energy and viral keywords."""
        scores = {}
        
        # Audio Energy
        try:
            y, sr = librosa.load(audio_path, sr=22050, offset=start, duration=duration)
            rms = librosa.feature.rms(y=y)[0]
            scores['audio'] = min(1.0, np.mean(rms) * 100)
        except:
            scores['audio'] = 0.5

        # Viral Keywords
        viral_words = ['secret', 'amazing', 'shocking', 'unbelievable', 'proven', 'hack', 'trick', 'truth', 'finally', 'insane']
        scores['keywords'] = min(1.0, sum(1 for w in viral_words if w in text.lower()) / 3)

        return (scores['audio'] * 0.4) + (scores['keywords'] * 0.6)

    def find_best_clips(self, duration: float, audio_path: str, words: List[Dict], num_clips: int) -> List[Dict]:
        logger.info(f"🧠 Analyzing for {num_clips} clips...")
        
        candidates = []
        i = 0
        target_dur = 45
        
        while i < len(words):
            if words[i]['start'] > duration - 20: break
            
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
                
                score = self.score_segment(audio_path, words[i]['start'], dur, transcript)
                
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
        prompt = f"""Generate viral TikTok content for: "{transcript[:500]}"
Niche: {niche}
Return JSON: {{ "title": "catchy title with emoji (max 50 chars)", "hashtags": ["#tag1", "#tag2", ...7 total] }}"""
        
        raw = call_gemini(prompt, self.config.GEMINI_API_KEY)
        try:
            return json.loads(raw.replace('```json','').replace('```','').strip())
        except: 
            return {"title": "Viral Short 🔥", "hashtags": ["#shorts", "#fyp", "#viral"]}

# ==================== VIRAL ANALYZER ====================

class ViralAnalyzer:
    def __init__(self, config: Config):
        self.config = config
    
    def analyze_hook(self, words: List[Dict], transcript: str) -> Dict:
        first_5_sec = [w for w in words if w['end'] <= 5]
        hook_text = ' '.join([w['text'] for w in first_5_sec])
        
        if not hook_text.strip():
            return {'score': 0.1, 'text': ''}
        
        prompt = f"""Rate this TikTok opening (first 5 seconds) for viral potential:
"{hook_text}"
Score 1-10 on curiosity, pattern interrupt, emotion. Return JSON: {{"score": 1-10}}"""
        
        raw = call_gemini(prompt, self.config.GEMINI_API_KEY)
        try:
            result = json.loads(re.search(r'\{[^{}]*\}', raw, re.DOTALL).group())
            return {'score': min(1.0, result.get('score', 5) / 10), 'text': hook_text}
        except:
            return {'score': 0.5, 'text': hook_text}
    
    def calculate_viral_score(self, clip: Dict, hook_analysis: Dict, audio_path: str) -> Dict:
        scores = {}
        
        scores['hook'] = hook_analysis.get('score', 0.5) * 30
        scores['content'] = clip.get('score', 0.5) * 30
        
        transcript = clip.get('transcript', '')
        duration = clip.get('duration', 45)
        words_per_min = (len(transcript.split()) / duration) * 60 if duration > 0 else 0
        
        if 140 <= words_per_min <= 180:
            scores['pacing'] = 25
        elif 120 <= words_per_min <= 200:
            scores['pacing'] = 18
        else:
            scores['pacing'] = 10
        
        try:
            y, sr = librosa.load(audio_path, sr=22050, offset=clip['start'], duration=min(duration, 60))
            rms = librosa.feature.rms(y=y)[0]
            scores['dynamics'] = min(1.0, np.std(rms) * 50) * 15
        except:
            scores['dynamics'] = 7.5
        
        total = sum(scores.values())
        
        if total >= 70:
            tier = "🔥 HIGH VIRAL"
        elif total >= 50:
            tier = "✅ GOOD"
        elif total >= 35:
            tier = "⚠️ MODERATE"
        else:
            tier = "❌ LOW"
        
        return {'total_score': round(total, 1), 'tier': tier}
    
    def get_trending_hashtags(self, niche: str, transcript: str) -> List[str]:
        prompt = f"""Generate 7 trending TikTok hashtags for:
"{transcript[:300]}"
Niche: {niche}
Return JSON array only: ["#tag1", ...]"""

        raw = call_gemini(prompt, self.config.GEMINI_API_KEY)
        try:
            hashtags = json.loads(raw.replace('```json','').replace('```','').strip())
            return [h if h.startswith('#') else f'#{h}' for h in hashtags[:7]]
        except:
            return ['#fyp', '#viral', '#shorts']

# ==================== VIDEO PROCESSOR ====================

class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config

    def download_video(self, url: str) -> Optional[str]:
        """Download video using POT plugin for YouTube bypass."""
        video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
        video_id = video_id_match.group(1) if video_id_match else "temp"
        path = f"{self.config.TEMP_FOLDER}/video_{video_id}.mp4"
        if os.path.exists(path): os.remove(path)
        
        opts = {
            'format': 'bv*+ba/b',  # Best video + best audio, fallback to best
            'outtmpl': path.replace('.mp4', '') + '.%(ext)s',
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }
        
        if os.path.exists(self.config.COOKIE_FILE):
            opts['cookiefile'] = self.config.COOKIE_FILE

        try:
            logger.info(f"⬇️ Downloading video {video_id}...")
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            
            # Find downloaded file (extension may vary)
            import glob
            downloaded_files = glob.glob(path.replace('.mp4', '') + '.*')
            if downloaded_files:
                actual_path = downloaded_files[0]
                if not actual_path.endswith('.mp4'):
                    import shutil
                    shutil.move(actual_path, path)
                    actual_path = path
                logger.info(f"✅ Downloaded: {os.path.getsize(actual_path) / (1024*1024):.1f} MB")
                return actual_path
            return None
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return None

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True, text=True, timeout=30
            )
            return float(result.stdout.strip())
        except:
            return 0

    def extract_audio(self, video_path: str) -> str:
        audio_path = f'{self.config.TEMP_FOLDER}/audio_{uuid.uuid4().hex[:6]}.wav'
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-ar', '16000', '-ac', '1', audio_path], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path

    def process_clip(self, video_path: str, clip: Dict, index: int) -> str:
        """Process clip with letterbox (black bars) layout."""
        start, dur = clip['start'], clip['duration']
        target_w, target_h = 1080, 1920
        out_file = os.path.join(self.config.OUTPUT_FOLDER, f"clip_{index}_{uuid.uuid4().hex[:4]}.mp4")
        ass_file = os.path.join(self.config.TEMP_FOLDER, f"sub_{index}.ass")

        # Create subtitles
        rel_words = [{'text': w['text'], 'start': w['start']-start, 'end': w['end']-start} for w in clip['words']]
        self._write_ass(rel_words, ass_file, target_w, target_h)

        # Letterbox layout (black bars)
        vf = f"[0:v]scale={target_w}:-1:flags=fast_bilinear,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,ass={ass_file}[outv]"

        cmd = ['ffmpeg', '-y', '-ss', str(start), '-t', str(dur), '-i', video_path, '-filter_complex', vf, 
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
        msg = ("🤖 <b>v10 ULTRA - Railway Edition</b>\n\n"
               "<b>Command:</b>\n/auto &lt;niche&gt;\n\n"
               "<b>Example:</b>\n<code>/auto tech podcasts</code>")
        await update.message.reply_text(msg, parse_mode='HTML')

    async def auto_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if not context.args:
            await update.message.reply_text("⚠️ Usage: /auto <query>")
            return
        
        query = " ".join(context.args)
        await update.message.reply_text(f"🔎 Searching for: '{query}'...")
        
        loop = asyncio.get_running_loop()
        videos = await loop.run_in_executor(None, self.finder.find_videos, query, 2)
        
        if not videos: 
            return await update.message.reply_text("❌ No videos found.")
        
        self.session_data[chat_id] = {'videos': videos, 'niche': query}
        
        msg = "📹 <b>Found Videos</b>\n\n"
        for i, v in enumerate(videos):
            msg += f"{i+1}. {v['title'][:50]}...\n   ⏱ {v['duration']//60} min\n\n"
        
        keyboard = [[InlineKeyboardButton("✅ Approve", callback_data="approve")],
                    [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]]
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='HTML')

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        chat_id = update.effective_chat.id
        await query.answer()
        
        if query.data == "cancel":
            await query.edit_message_text("❌ Cancelled.")
            self.session_data.pop(chat_id, None)
            return
            
        if query.data == "approve":
            if chat_id not in self.session_data:
                await query.edit_message_text("⚠️ Session expired.")
                return
            self.user_states[chat_id] = "WAITING_CLIP_COUNT"
            await query.edit_message_text("🔢 <b>How many clips?</b>\nReply 1-20", parse_mode='HTML')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if self.user_states.get(chat_id) == "WAITING_CLIP_COUNT":
            try:
                count = int(update.message.text.strip())
                if not 1 <= count <= 20: raise ValueError
                
                self.session_data[chat_id]['clip_count'] = count
                self.user_states[chat_id] = None
                
                await update.message.reply_text(f"🚀 Processing {count} clips...")
                asyncio.create_task(self.run_processing(
                    self.session_data[chat_id]['videos'], 
                    count, 
                    update.message,
                    self.session_data[chat_id].get('niche', 'general')
                ))
            except ValueError:
                await update.message.reply_text("⚠️ Enter 1-20")

    async def run_processing(self, videos, total_clips, message, niche):
        loop = asyncio.get_running_loop()
        vp = VideoProcessor(self.config)
        ca = ContentAnalyzer(self.config)
        va = ViralAnalyzer(self.config)
        
        try:
            clips_per_video = max(1, total_clips // len(videos))
            clip_index = 0
            
            for vid_idx, video in enumerate(videos):
                remaining = total_clips - clip_index
                if remaining <= 0: break
                
                target = min(clips_per_video, remaining)
                if vid_idx == len(videos) - 1: target = remaining
                
                await message.reply_text(f"📺 Video {vid_idx+1}/{len(videos)}: {video['title'][:40]}...")
                
                vpath = await loop.run_in_executor(None, vp.download_video, video['url'])
                if not vpath: 
                    await message.reply_text("❌ Download failed")
                    continue
                
                duration = await loop.run_in_executor(None, vp.get_video_duration, vpath)
                apath = await loop.run_in_executor(None, vp.extract_audio, vpath)
                words = await loop.run_in_executor(None, ca.transcribe_audio, apath)
                
                if len(words) > 100:
                    await message.reply_text("🧠 Analyzing...")
                    clips = await loop.run_in_executor(None, ca.find_best_clips, duration, apath, words, target)
                    
                    for clip in clips:
                        clip_index += 1
                        rel_words = [{'text': w['text'], 'start': w['start']-clip['start'], 'end': w['end']-clip['start']} for w in clip['words']]
                        
                        hook = await loop.run_in_executor(None, va.analyze_hook, rel_words, clip['transcript'])
                        viral = await loop.run_in_executor(None, va.calculate_viral_score, clip, hook, apath)
                        out = await loop.run_in_executor(None, vp.process_clip, vpath, clip, clip_index)
                        seo = await loop.run_in_executor(None, ca.generate_seo, clip['transcript'], niche)
                        tags = await loop.run_in_executor(None, va.get_trending_hashtags, niche, clip['transcript'])
                        
                        all_tags = list(dict.fromkeys(seo.get('hashtags', []) + tags))[:8]
                        caption = f"📊 {viral['tier']} ({viral['total_score']}/100)\n\n📝 {seo['title']}\n\n{' '.join(all_tags)}"
                        
                        if os.path.exists(out):
                            with open(out, 'rb') as f:
                                await message.reply_video(video=f, caption=caption[:1024])
                            os.remove(out)
                
                if os.path.exists(vpath): os.remove(vpath)
                if os.path.exists(apath): os.remove(apath)
                
            await message.reply_text(f"🏁 Done! {clip_index} clips sent.")
                
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
            logger.error("Error", exc_info=True)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log the error and handle specific cases."""
        logger.error(f"Exception while handling an update: {context.error}")
        
    def run(self):
        logger.info("🤖 Starting bot...")
        self.app.add_error_handler(self.error_handler)
        # Drop pending updates to flush old conflicting messages
        self.app.run_polling(drop_pending_updates=True, close_loop=False)

# ==================== MAIN ====================

if __name__ == "__main__":
    bot = BotInterface(CONFIG)
    bot.run()
