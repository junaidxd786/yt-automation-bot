# YouTube Shorts Generator - Railway Edition

AI-powered YouTube Shorts generator that runs on Railway.

## Features

- 🧠 **Deep ML Scoring**: Audio + Sentiment + Viral Keywords
- 🎣 **Hook Detection**: AI analyzes first 5 seconds
- 📊 **Viral Score**: Predicts viral potential (0-100)
- 🏷️ **Trending Hashtags**: Niche-aware hashtag generation
- 📝 **Auto Subtitles**: ASS subtitles burned in
- 👤 **Smart Crop**: YOLO person tracking

## Quick Deploy to Railway

1. **Fork/Clone this repo**

2. **Create Railway Project**
   - Go to [railway.app](https://railway.app)
   - New Project → Deploy from GitHub
   - Select this repo

3. **Add Environment Variables**
   In Railway dashboard → Variables:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   DEEPGRAM_API_KEY=your_deepgram_key
   GEMINI_API_KEY=your_gemini_key
   ```

4. **Deploy** - Railway auto-builds from Dockerfile

## Local Development

```bash
# Clone
git clone <your-repo>
cd youtube_shorts_railway

# Create .env file
cp .env.example .env
# Edit .env with your keys

# Build & Run
docker build -t shorts-bot .
docker run --env-file .env shorts-bot
```

## Usage

1. Open Telegram and message your bot
2. Send `/start` to see commands
3. Send `/auto tech podcasts` to find videos
4. Approve videos and enter clip count
5. Clips are sent directly to Telegram!

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | From @BotFather |
| `TELEGRAM_CHAT_ID` | Your chat ID |
| `DEEPGRAM_API_KEY` | For transcription |
| `GEMINI_API_KEY` | For SEO & hook analysis |

## Cost Estimates

- **Railway**: ~$5/month (hobby plan)
- **Deepgram**: Free tier = 12,500 mins/month
- **Gemini**: Free tier = 60 req/min

---

Built for Railway deployment. Original Colab version: v10 Ultra.
