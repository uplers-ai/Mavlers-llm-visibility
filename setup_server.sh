#!/bin/bash
# ==============================================================================
# LLM Visibility Audit Tool - Server Setup Script
# ==============================================================================
# This script automates the server setup process.
# Run on a fresh Ubuntu 22.04 server.
#
# Usage: bash setup_server.sh
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$HOME/visibility_audit"
VENV_DIR="$PROJECT_DIR/venv"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     LLM Visibility Audit Tool - Server Setup                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ------------------------------------------------------------------------------
# Step 1: Update System
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[1/7] Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# ------------------------------------------------------------------------------
# Step 2: Install System Dependencies
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[2/7] Installing system dependencies...${NC}"
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0

# ------------------------------------------------------------------------------
# Step 3: Create Project Directory
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[3/7] Creating project directory...${NC}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# Step 4: Set Up Python Virtual Environment
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[4/7] Setting up Python virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------------------------
# Step 5: Install Python Dependencies
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[5/7] Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install openai anthropic google-genai requests python-dotenv playwright

# ------------------------------------------------------------------------------
# Step 6: Install Playwright Browser
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[6/7] Installing Playwright Chromium browser...${NC}"
playwright install chromium

# ------------------------------------------------------------------------------
# Step 7: Create .env Template
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[7/7] Creating configuration template...${NC}"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    cat > "$PROJECT_DIR/.env" << 'EOF'
# LLM Visibility Audit Tool - Configuration
# Fill in your API keys below

# ============================================================================
# LLM API KEYS (set at least one)
# ============================================================================
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
XAI_API_KEY=
PERPLEXITY_API_KEY=

# ============================================================================
# EMAIL NOTIFICATIONS (optional but recommended)
# For Gmail: Use App Password - https://support.google.com/accounts/answer/185833
# ============================================================================
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
EMAIL_TO=

# ============================================================================
# ARCHIVE SETTINGS
# ============================================================================
ARCHIVE_RETENTION_WEEKS=12
EOF
    chmod 600 "$PROJECT_DIR/.env"
    echo -e "${GREEN}Created .env template at $PROJECT_DIR/.env${NC}"
else
    echo -e "${YELLOW}.env file already exists, skipping...${NC}"
fi

# ------------------------------------------------------------------------------
# Create Cron Setup Helper Script
# ------------------------------------------------------------------------------
cat > "$PROJECT_DIR/setup_cron.sh" << 'EOF'
#!/bin/bash
# Add weekly cron job for visibility audit

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CRON_CMD="0 6 * * 0 cd $SCRIPT_DIR && $SCRIPT_DIR/venv/bin/python visibility_audit2.0.py >> $SCRIPT_DIR/cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "visibility_audit2.0.py"; then
    echo "Cron job already exists!"
    crontab -l | grep "visibility_audit2.0.py"
else
    # Add cron job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Cron job added! Will run every Sunday at 6:00 AM"
    echo "Command: $CRON_CMD"
fi
EOF
chmod +x "$PROJECT_DIR/setup_cron.sh"

# ------------------------------------------------------------------------------
# Print Success Message
# ------------------------------------------------------------------------------
echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE! ✅                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Upload the visibility_audit2.0.py script to: $PROJECT_DIR/"
echo ""
echo "2. Edit the .env file with your API keys:"
echo "   ${YELLOW}nano $PROJECT_DIR/.env${NC}"
echo ""
echo "3. Test the script:"
echo "   ${YELLOW}cd $PROJECT_DIR && source venv/bin/activate${NC}"
echo "   ${YELLOW}python visibility_audit2.0.py --test-email${NC}"
echo "   ${YELLOW}python visibility_audit2.0.py${NC}"
echo ""
echo "4. Set up weekly cron job:"
echo "   ${YELLOW}$PROJECT_DIR/setup_cron.sh${NC}"
echo ""
echo -e "${GREEN}Project directory: $PROJECT_DIR${NC}"
echo ""

