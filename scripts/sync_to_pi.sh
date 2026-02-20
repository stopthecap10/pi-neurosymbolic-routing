#!/bin/bash
# Sync code to Raspberry Pi

PI_HOST="stopthecap10@192.168.1.179"
PI_DIR="~/pi-neurosymbolic-routing"

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "⚠️  sshpass is not installed. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install hudochenkov/sshpass/sshpass
    else
        echo "❌ Homebrew not found. Please install sshpass manually:"
        echo "   brew install hudochenkov/sshpass/sshpass"
        exit 1
    fi
fi

# Prompt for password (hidden input)
echo "🔐 Enter password for ${PI_HOST}:"
read -s PI_PASSWORD
echo ""

echo "📤 Syncing code to Pi at ${PI_HOST}..."

sshpass -p "${PI_PASSWORD}" rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no" \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='outputs/' \
    --exclude='results/' \
    --exclude='runs_*/' \
    --exclude='artifacts/' \
    --exclude='archive/' \
    --exclude='.claude/' \
    --exclude='data/cache/' \
    ./ ${PI_HOST}:${PI_DIR}/

if [ $? -eq 0 ]; then
    echo "✅ Sync complete!"
    echo ""
    echo "To SSH into Pi, run:"
    echo "  sshpass -p 'YOUR_PASSWORD' ssh ${PI_HOST}"
    echo ""
    echo "To setup on Pi (first time), run:"
    echo "  sshpass -p 'YOUR_PASSWORD' ssh ${PI_HOST} 'cd ${PI_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install -e .'"
else
    echo "❌ Sync failed. Please check your password and network connection."
    exit 1
fi
