# Build v6 - With PO Token Provider for YouTube bypass
FROM python:3.11-slim

# Install system dependencies including Node.js for POT provider
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone and build the POT provider server
RUN git clone --single-branch --branch 1.2.2 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /opt/pot-provider \
    && cd /opt/pot-provider/server \
    && npm install \
    && npx tsc

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the POT provider plugin for yt-dlp
RUN pip install --no-cache-dir bgutil-ytdlp-pot-provider

COPY . .

RUN mkdir -p /app/data/temp_processing /app/data/output

# Create startup script that runs POT server in background + main app
RUN echo '#!/bin/bash\ncd /opt/pot-provider/server && node build/main.js &\nsleep 3\npython /app/main.py' > /app/start.sh && chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]
