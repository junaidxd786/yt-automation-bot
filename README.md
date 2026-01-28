# 🎬 YouTube Shorts Generator

AI-powered tool that automatically finds viral moments in long-form YouTube videos and converts them into engaging short-form content.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep ML Scoring** | Analyzes audio energy, sentiment, and viral keywords |
| 🎣 **Hook Detection** | AI evaluates first 5 seconds for engagement potential |
| 📊 **Viral Prediction** | Scores clips 0-100 based on multiple factors |
| 🏷️ **Smart Hashtags** | Generates niche-aware trending hashtags |
| 📝 **Auto Subtitles** | Burns in word-by-word captions |
| 👤 **Smart Cropping** | YOLO-based person tracking for vertical format |

## 🛠️ Tech Stack

- **ML Models**: RoBERTa (sentiment), YOLOv8 (person detection)
- **Transcription**: Deepgram Nova-2
- **Content AI**: Google Gemini 2.0
- **Video Processing**: FFmpeg with hardware acceleration
- **Interface**: Telegram Bot API

## 📸 How It Works

```
1. /auto tech podcasts     → Search YouTube
2. Approve found videos    → Select what to process
3. Enter clip count        → How many shorts to create
4. AI analyzes content     → Finds best moments
5. Receive finished clips  → Sent to Telegram
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (recommended)
- API Keys: Telegram Bot, Deepgram, Gemini

### Run with Docker

```bash
docker build -t shorts-generator .
docker run --env-file .env shorts-generator
```

### Environment Variables

Create a `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
DEEPGRAM_API_KEY=your_key
GEMINI_API_KEY=your_key
```

## 📁 Project Structure

```
├── main.py           # Core application
├── Dockerfile        # Container configuration
├── requirements.txt  # Python dependencies
└── data/             # Runtime data (gitignored)
```

## 📄 License

MIT License - feel free to use and modify.

---

<p align="center">
  <b>Transform long videos into viral shorts with AI</b>
</p>
