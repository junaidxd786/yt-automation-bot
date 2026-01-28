# YouTube Shorts Generator - Cloud Run Deployment

## Quick Deploy to Google Cloud Run

### Prerequisites
1. [Google Cloud account](https://cloud.google.com/) with billing enabled
2. [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed

### Step 1: Set Up Project
```bash
# Login to GCP
gcloud auth login

# Create project (or use existing)
gcloud projects create youtube-shorts-bot --name="YouTube Shorts Bot"
gcloud config set project youtube-shorts-bot

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Step 2: Deploy
```bash
cd youtube_shorts_railway

# Deploy to Cloud Run (this builds and deploys in one command)
gcloud run deploy youtube-shorts-bot \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "TELEGRAM_BOT_TOKEN=your_token" \
  --set-env-vars "TELEGRAM_CHAT_ID=your_chat_id" \
  --set-env-vars "DEEPGRAM_API_KEY=your_key" \
  --set-env-vars "GEMINI_API_KEY=your_key" \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --min-instances 1
```

### Step 3: Verify
- Check Cloud Run console: https://console.cloud.google.com/run
- Test Telegram bot: Send `/start`

## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| TELEGRAM_BOT_TOKEN | ✅ | Telegram Bot API token |
| TELEGRAM_CHAT_ID | ❌ | Your Telegram chat ID |
| DEEPGRAM_API_KEY | ✅ | For audio transcription |
| GEMINI_API_KEY | ✅ | For AI analysis |
| YOUTUBE_COOKIES | ❌ | JSON cookies (optional, try without first) |

## Cost Estimate
- **Free tier:** 2 million requests/month free
- **Typical:** $0-5/month for light usage
- **min-instances 1** keeps bot always running (~$10-15/month)

## Why Cloud Run?
- Uses Google's infrastructure (YouTube trusts Google IPs)
- Same Docker deployment as Railway
- Scales automatically
- Built-in logging
