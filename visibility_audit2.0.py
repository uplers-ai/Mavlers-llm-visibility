#!/usr/bin/env python3
"""
LLM Visibility Audit Tool for Uplers
=====================================
Measures brand visibility for Uplers (AI-powered talent platform connecting 
global companies with pre-vetted developers from India) across multiple LLMs:
- ChatGPT (OpenAI)
- Claude (Anthropic)
- Gemini (Google)
- Grok (xAI)
- Perplexity

Features:
- Runs 50 prompts across 10 intent categories, 3x each for statistical reliability
- Generates HTML dashboard with visibility scores and competitor rankings
- Automatic screenshot capture of dashboard
- Email notifications with summary
- Historical comparison (week-over-week changes)
- Timestamped archives with configurable retention
- Error recovery with automatic retries
- Designed for weekly automated execution via cron

Usage:
    python visibility_audit2.0.py                    # Run audit
    python visibility_audit2.0.py --test-email      # Test email configuration
    python visibility_audit2.0.py --setup-cron      # Show cron setup instructions

Requirements:
    pip install openai anthropic google-generativeai requests python-dotenv playwright --break-system-packages
    playwright install chromium

Environment Variables (set in .env file or export):
    # LLM API Keys (set at least one)
    OPENAI_API_KEY=your_openai_key
    ANTHROPIC_API_KEY=your_anthropic_key
    GOOGLE_API_KEY=your_google_gemini_key
    XAI_API_KEY=your_xai_grok_key
    PERPLEXITY_API_KEY=your_perplexity_key
    
    # Email Configuration (optional, for notifications)
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your_email@gmail.com
    SMTP_PASSWORD=your_app_password
    EMAIL_TO=recipient@example.com
    
    # Archive Settings (optional)
    ARCHIVE_RETENTION_WEEKS=12
"""

import os
import sys
import json
import re
import time
import glob
import shutil
import smtplib
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import traceback
import requests

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visibility_audit.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# API Clients (will be initialized after checking keys)
openai_client = None
anthropic_client = None
gemini_model = None
xai_api_key = None
perplexity_api_key = None

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_COMPANY = "Uplers"
TARGET_REGION = "USA"
RUNS_PER_PROMPT = 3

# Archive settings
ARCHIVE_DIR = "archives"
ARCHIVE_RETENTION_WEEKS = int(os.getenv("ARCHIVE_RETENTION_WEEKS", "12"))

# Retry settings for error recovery
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Email settings
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")
EMAIL_ENABLED = bool(SMTP_USER and SMTP_PASSWORD and EMAIL_TO)

# LLM Enable/Disable settings (set to "false" to disable)
# Grok is disabled by default due to severe rate limits on free tier
ENABLE_CHATGPT = os.getenv("ENABLE_CHATGPT", "true").lower() == "true"
ENABLE_CLAUDE = os.getenv("ENABLE_CLAUDE", "true").lower() == "true"
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "true").lower() == "true"
ENABLE_GROK = os.getenv("ENABLE_GROK", "true").lower() == "true"  # Enabled - paid API
ENABLE_PERPLEXITY = os.getenv("ENABLE_PERPLEXITY", "true").lower() == "true"  # Enabled - paid API

# Auto-skip settings: Skip an LLM if it fails this many times consecutively
CONSECUTIVE_FAILURES_TO_SKIP = 3

# Track consecutive failures per LLM
llm_failure_counts = {}

# Known platform patterns to help with detection (will also find new ones dynamically)
KNOWN_PLATFORMS = [
    "Uplers", "Toptal", "Turing", "Andela", "Arc", "CloudDevs", 
    "Terminal", "Gun.io", "Lemon.io", "BairesDev", "Revelo",
    "Supersourcing", "Gigster", "Multiplier", "Remote", "Deel", 
    "Oyster", "Globalization Partners", "Fiverr", "Upwork", "Freelancer",
    "LinkedIn", "Indeed", "Hired", "Triplebyte", "Vettery", "Crossover",
    "X-Team", "Scalable Path", "Codementor", "RemoteOK", "We Work Remotely",
    "AngelList", "Stack Overflow", "GitHub Jobs", "Dice", "Naukri",
    "TalentScale", "Flexiple", "RemotePanda", "HackerRank", "CodeSignal"
]

# Intent categories with prompts
PROMPTS_BY_INTENT = {
    "Top Platforms for Indian Developers": [
        "What are the top platforms for tech companies to hire vetted Indian software engineers?",
        "Best platform to hire Indian developers",
        "What is the best platform to hire Indian developers",
        "Top talent platforms India comparison",
        "Where can I find vetted Indian software engineers",
    ],
    "Competitors and Comparisons": [
        "Who are the top competitors in hiring vetted software engineers from India for tech companies?",
        "Who are the top competitors in AI-driven recruitment for tech companies hiring vetted software engineers and marketing talent from India?",
    ],
    "AI-Driven Recruitment": [
        "AI-driven recruitment platform to hire Software Engineers",
        "AI-driven recruitment platform to hire remote Engineers",
        "AI-powered hiring platform",
        "AI Vetted Indian tech talent",
    ],
    "Hiring Managers Multi-Role": [
        "What are the top platforms for hiring managers looking to recruit vetted Indian software engineers and marketing professionals?",
    ],
    "How to Hire from India": [
        "How to hire developers from India",
        "How to find vetted developers India",
        "How to hire remote developers quickly",
        "Steps to hire offshore developers",
        "How to interview Indian developers",
        "Who can help me hire remote developers from India",
    ],
    "Direct Hiring Queries": [
        "Hire software engineers from India",
        "Hire remote developers India",
        "Hire full stack developers India",
        "Hire AI/ML engineers India",
        "Hire data scientists India",
        "Hire DevOps engineers India",
        "Hire QA engineers India",
        "Hire solutions architects India",
        "Hire marketing professionals India",
        "Vetted Indian developers",
    ],
    "Talent Platform for Tech Companies": [
        "Talent platform for tech companies",
        "Hire developers for SaaS companies",
        "Hire engineers for Startup companies",
        "Hire Engineers for Tech Startups",
        "Tech talent for startups",
        "Indian talent for global companies",
    ],
    "Cost and Budget": [
        "Cost of hiring developers from India",
        "Budget-friendly tech hiring",
        "How much does it cost to hire a developer from India",
        "India salary guide for tech hiring",
    ],
    "Fast Hiring Process": [
        "Fast hiring process for developers",
        "How to hire remote developers quickly",
    ],
    "Vetting Process": [
        "Vetting process for Indian developers",
        "Where can I find vetted Indian software engineers",
    ],
    "Hiring Models": [
        "Staff augmentation services India",
        "Flexible hiring models India",
        "Contract to hire developers India",
        "Full time hiring platforms",
    ],
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def retry_with_backoff(func, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Decorator for retrying functions with exponential backoff."""
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Longer wait for rate limit errors (429)
                if "429" in str(e) or "rate" in error_str or "exhausted" in error_str or "quota" in error_str:
                    wait_time = 30 * (2 ** attempt)  # Start with 30s for rate limits
                    logger.warning(f"Rate limit hit! Attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s...")
                else:
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                
                time.sleep(wait_time)
        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {last_exception}")
        return ""
    return wrapper


def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y-%m-%d")


def get_week_number():
    """Get ISO week number for weekly tracking."""
    return datetime.now().strftime("%Y-W%V")


def ensure_archive_dir():
    """Create archive directory if it doesn't exist."""
    Path(ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)


def cleanup_old_archives():
    """Remove archives older than retention period."""
    ensure_archive_dir()
    cutoff_date = datetime.now() - timedelta(weeks=ARCHIVE_RETENTION_WEEKS)
    
    removed_count = 0
    for filepath in glob.glob(f"{ARCHIVE_DIR}/*"):
        try:
            # Extract date from filename
            filename = os.path.basename(filepath)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                file_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                if file_date < cutoff_date:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    else:
                        shutil.rmtree(filepath)
                    removed_count += 1
                    logger.info(f"Removed old archive: {filename}")
        except Exception as e:
            logger.warning(f"Error processing archive {filepath}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} old archive(s)")


def get_previous_results():
    """Load the most recent previous audit results for comparison."""
    ensure_archive_dir()
    
    # Find all previous result files
    result_files = sorted(glob.glob(f"{ARCHIVE_DIR}/audit_results_*.json"), reverse=True)
    
    if len(result_files) >= 1:
        # Get the most recent one (skip current if it exists)
        for filepath in result_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Extract date from filename
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(filepath))
                    if date_match:
                        data['_archive_date'] = date_match.group(1)
                    return data
            except Exception as e:
                logger.warning(f"Could not load {filepath}: {e}")
                continue
    
    return None


# ============================================================================
# API CLIENT INITIALIZATION
# ============================================================================

def initialize_clients():
    """Initialize API clients based on available keys."""
    global openai_client, anthropic_client, gemini_model, xai_api_key, perplexity_api_key
    
    clients_available = []
    
    # OpenAI (ChatGPT)
    if os.getenv("OPENAI_API_KEY") and ENABLE_CHATGPT:
        try:
            from openai import OpenAI
            openai_client = OpenAI()
            clients_available.append("OpenAI")
            logger.info("✅ OpenAI client initialized")
        except ImportError:
            logger.warning("⚠️  OpenAI package not installed. Run: pip install openai")
    elif os.getenv("OPENAI_API_KEY") and not ENABLE_CHATGPT:
        logger.info("⏭️  ChatGPT disabled (ENABLE_CHATGPT=false)")
    
    # Anthropic (Claude)
    if os.getenv("ANTHROPIC_API_KEY") and ENABLE_CLAUDE:
        try:
            import anthropic
            anthropic_client = anthropic.Anthropic()
            clients_available.append("Anthropic")
            logger.info("✅ Anthropic client initialized")
        except ImportError:
            logger.warning("⚠️  Anthropic package not installed. Run: pip install anthropic")
    elif os.getenv("ANTHROPIC_API_KEY") and not ENABLE_CLAUDE:
        logger.info("⏭️  Claude disabled (ENABLE_CLAUDE=false)")
    
    # Google (Gemini)
    if os.getenv("GOOGLE_API_KEY") and ENABLE_GEMINI:
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            clients_available.append("Google")
            logger.info("✅ Google Gemini client initialized")
        except ImportError:
            logger.warning("⚠️  Google GenAI package not installed. Run: pip install google-generativeai")
    elif os.getenv("GOOGLE_API_KEY") and not ENABLE_GEMINI:
        logger.info("⏭️  Gemini disabled (ENABLE_GEMINI=false)")
    
    # xAI (Grok) - Disabled by default due to severe rate limits
    if os.getenv("XAI_API_KEY") and ENABLE_GROK:
        xai_api_key = os.getenv("XAI_API_KEY")
        clients_available.append("xAI")
        logger.info("✅ xAI Grok client initialized")
    elif os.getenv("XAI_API_KEY") and not ENABLE_GROK:
        logger.info("⏭️  xAI Grok disabled (ENABLE_GROK=false) - severe rate limits on free tier")
    
    # Perplexity
    if os.getenv("PERPLEXITY_API_KEY") and ENABLE_PERPLEXITY:
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        clients_available.append("Perplexity")
        logger.info("✅ Perplexity client initialized")
    elif os.getenv("PERPLEXITY_API_KEY") and not ENABLE_PERPLEXITY:
        logger.info("⏭️  Perplexity disabled (ENABLE_PERPLEXITY=false)")
    
    return clients_available


# ============================================================================
# LLM QUERY FUNCTIONS (with retry logic)
# ============================================================================

def query_openai(prompt: str) -> str:
    """Query OpenAI GPT-4.1 with retry logic."""
    if not openai_client:
        return ""
    
    def _query():
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. The user is based in {TARGET_REGION}. When recommending platforms or companies, please be specific and name them."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    return retry_with_backoff(_query)()


def query_anthropic(prompt: str) -> str:
    """Query Anthropic Claude with retry logic."""
    if not anthropic_client:
        return ""
    
    def _query():
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=f"You are a helpful assistant. The user is based in {TARGET_REGION}. When recommending platforms or companies, please be specific and name them.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    return retry_with_backoff(_query)()


def query_gemini(prompt: str) -> str:
    """Query Google Gemini with retry logic."""
    if not gemini_model:
        return ""
    
    def _query():
        # Add delay to avoid rate limiting (Google free tier has strict limits)
        time.sleep(2)
        full_prompt = f"You are a helpful assistant. The user is based in {TARGET_REGION}. When recommending platforms or companies, please be specific and name them.\n\nUser: {prompt}"
        response = gemini_model.generate_content(full_prompt)
        return response.text
    
    return retry_with_backoff(_query)()


def query_grok(prompt: str) -> str:
    """Query xAI Grok with retry logic."""
    if not xai_api_key:
        return ""
    
    def _query():
        headers = {
            "Authorization": f"Bearer {xai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-3-latest",
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant. The user is based in {TARGET_REGION}. When recommending platforms or companies, please be specific and name them."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=120  # Increased timeout for Grok API
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    return retry_with_backoff(_query)()


def query_perplexity(prompt: str) -> str:
    """Query Perplexity AI with retry logic."""
    if not perplexity_api_key:
        return ""
    
    def _query():
        headers = {
            "Authorization": f"Bearer {perplexity_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant. The user is based in {TARGET_REGION}. When recommending platforms or companies, please be specific and name them."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    return retry_with_backoff(_query)()

def extract_companies(text: str) -> dict:
    """Extract company/platform mentions from text dynamically."""
    if not text:
        return {}
    
    mentions = {}
    text_lower = text.lower()
    
    # 1. First check known platforms with variations
    platform_variations = {
        "Uplers": ["uplers", "uplers.com", "uplers AI hiring platform", "uplers ai", "uplers platform", "uplers talent"],
        "Toptal": ["toptal", "toptal.com"],
        "Turing": ["turing", "turing.com"],
        "Andela": ["andela", "andela.com"],
        "Arc": ["arc.dev", "arc dev"],
        "CloudDevs": ["clouddevs", "cloud devs", "clouddevs.com"],
        "Terminal": ["terminal.io", "terminal io"],
        "Gun.io": ["gun.io", "gun io", "gunio"],
        "Lemon.io": ["lemon.io", "lemon io", "lemonpicker"],
        "BairesDev": ["bairesdev", "baires dev", "bairesdev.com"],
        "Revelo": ["revelo", "revelo.com"],
        "Supersourcing": ["supersourcing", "super sourcing"],
        "Gigster": ["gigster", "gigster.com"],
        "Multiplier": ["multiplier", "multiplier.com"],
        "Remote": ["remote.com", "remote.co"],
        "Deel": ["deel", "deel.com"],
        "Oyster": ["oyster", "oysterhr", "oyster hr"],
        "Globalization Partners": ["globalization partners", "g-p.com", "g-p"],
        "Fiverr": ["fiverr", "fiverr.com"],
        "Upwork": ["upwork", "upwork.com"],
        "Freelancer": ["freelancer.com", "freelancer.in"],
        "LinkedIn": ["linkedin", "linkedin.com"],
        "Indeed": ["indeed", "indeed.com"],
        "Hired": ["hired.com", "hired platform"],
        "Triplebyte": ["triplebyte"],
        "Vettery": ["vettery"],
        "Crossover": ["crossover", "crossover.com"],
        "X-Team": ["x-team", "xteam"],
        "Scalable Path": ["scalable path", "scalablepath"],
        "Codementor": ["codementor", "codementorx"],
        "RemoteOK": ["remoteok", "remote ok"],
        "We Work Remotely": ["we work remotely", "weworkremotely"],
        "AngelList": ["angellist", "angel list", "wellfound"],
        "Stack Overflow Jobs": ["stack overflow jobs", "stackoverflow jobs"],
        "GitHub Jobs": ["github jobs"],
        "Dice": ["dice.com"],
        "Naukri": ["naukri", "naukri.com"],
        "TalentScale": ["talentscale", "talent scale"],
        "Flexiple": ["flexiple"],
        "RemotePanda": ["remotepanda", "remote panda"],
        "HackerRank": ["hackerrank", "hacker rank"],
        "CodeSignal": ["codesignal", "code signal"],
        "Karat": ["karat"],
        "Wework": ["wework"],
        "Talent500": ["talent500", "talent 500"],
        "Pesto": ["pesto.tech", "pesto tech"],
        "GeeksforGeeks": ["geeksforgeeks", "gfg jobs"],
        "Instahyre": ["instahyre"],
        "Hirect": ["hirect"],
        "Cutshort": ["cutshort"],
        "Hirist": ["hirist"],
        "iimjobs": ["iimjobs"],
        "Freshersworld": ["freshersworld"],
        "Shine": ["shine.com"],
        "Monster": ["monster.com", "monster india"],
        "Glassdoor": ["glassdoor"],
        "ZipRecruiter": ["ziprecruiter", "zip recruiter"],
        "SimplyHired": ["simplyhired", "simply hired"],
        "CareerBuilder": ["careerbuilder", "career builder"],
        "Snaphunt": ["snaphunt"],
        "Workable": ["workable"],
        "Lever": ["lever.co", "lever hiring"],
        "Greenhouse": ["greenhouse.io", "greenhouse"],
        "Recruiterflow": ["recruiterflow"],
        "Zoho Recruit": ["zoho recruit"],
        "Bamboo HR": ["bamboohr", "bamboo hr"],
        "JazzHR": ["jazzhr", "jazz hr"],
    }
    
    for platform, patterns in platform_variations.items():
        count = 0
        for pattern in patterns:
            pattern_lower = pattern.lower()
            # Use word boundaries for short patterns
            if len(pattern_lower) <= 4:
                count += len(re.findall(r'\b' + re.escape(pattern_lower) + r'\b', text_lower))
            else:
                count += text_lower.count(pattern_lower)
        
        if count > 0:
            mentions[platform] = count
    
    # 2. Also extract URLs/domains that might be platforms (*.io, *.com, *.dev)
    url_patterns = re.findall(r'\b([a-zA-Z][a-zA-Z0-9]*(?:\.[a-zA-Z0-9]+)*\.(?:io|com|dev|co|tech|ai))\b', text, re.IGNORECASE)
    for url in url_patterns:
        # Extract platform name from URL
        platform_name = url.split('.')[0].capitalize()
        if platform_name not in mentions and len(platform_name) > 2:
            # Check it's not already counted under a known name
            already_counted = False
            for known in mentions.keys():
                if platform_name.lower() in known.lower() or known.lower() in platform_name.lower():
                    already_counted = True
                    break
            if not already_counted:
                mentions[url] = 1
    
    # 3. Look for numbered list items that might be platform names (e.g., "1. Toptal", "- Uplers")
    list_patterns = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-•*]\s*)([A-Z][a-zA-Z0-9\.\-]+(?:\s+[A-Z][a-zA-Z0-9\.\-]+)?)', text)
    for item in list_patterns:
        item_clean = item.strip()
        if len(item_clean) > 2 and item_clean not in mentions:
            # Check if it looks like a company name (not common words)
            common_words = ['the', 'and', 'for', 'with', 'from', 'this', 'that', 'they', 'have', 'will', 'can', 'how', 'what', 'when', 'where', 'which', 'best', 'top', 'good', 'great', 'here', 'some', 'many', 'most', 'also', 'other', 'more', 'very', 'just', 'only', 'even', 'such', 'like', 'well', 'back', 'been', 'being', 'both', 'each', 'find', 'first', 'get', 'give', 'go', 'look', 'make', 'need', 'new', 'now', 'over', 'see', 'take', 'time', 'want', 'way', 'work', 'year', 'know', 'could', 'into', 'than', 'then', 'them', 'these', 'think', 'through', 'would', 'about', 'after', 'before', 'between', 'come', 'down', 'during', 'high', 'long', 'made', 'part', 'people', 'place', 'same', 'should', 'still', 'under', 'while', 'again', 'against', 'below', 'between', 'different', 'does', 'doing', 'done', 'enough', 'every', 'example', 'following', 'found', 'further', 'given', 'going', 'great', 'high', 'higher', 'however', 'important', 'including', 'large', 'later', 'less', 'little', 'local', 'looking', 'lower', 'major', 'making', 'must', 'never', 'number', 'often', 'open', 'possible', 'present', 'rather', 'recent', 'right', 'second', 'several', 'since', 'small', 'social', 'something', 'special', 'state', 'states', 'sure', 'system', 'things', 'those', 'three', 'today', 'together', 'trying', 'using', 'various', 'ways', 'within', 'without', 'working', 'world', 'years', 'young', 'software', 'developer', 'developers', 'engineer', 'engineers', 'platform', 'platforms', 'hiring', 'hire', 'talent', 'remote', 'india', 'indian', 'company', 'companies', 'based', 'services', 'service']
            if item_clean.lower() not in common_words:
                # Check it's not already a known platform
                already_counted = False
                for known in mentions.keys():
                    if item_clean.lower() == known.lower():
                        already_counted = True
                        break
                if not already_counted:
                    mentions[item_clean] = 1
    
    return mentions

def run_single_query(llm_name: str, prompt: str, intent: str, run_num: int):
    """Run a single query and return results."""
    global llm_failure_counts
    
    query_funcs = {
        "ChatGPT": query_openai,
        "Claude": query_anthropic,
        "Gemini": query_gemini,
        "Grok": query_grok,
        "Perplexity": query_perplexity
    }
    
    # Check if this LLM should be skipped due to consecutive failures
    if llm_failure_counts.get(llm_name, 0) >= CONSECUTIVE_FAILURES_TO_SKIP:
        # Return empty result - LLM is being skipped
        return {
            "llm": llm_name,
            "intent": intent,
            "prompt": prompt,
            "run": run_num,
            "response": "",
            "companies_mentioned": {},
            "target_mentioned": False,
            "timestamp": datetime.now().isoformat(),
            "skipped": True
        }
    
    response = query_funcs[llm_name](prompt)
    companies = extract_companies(response)
    
    # Track failures for auto-skip
    if not response:
        llm_failure_counts[llm_name] = llm_failure_counts.get(llm_name, 0) + 1
        if llm_failure_counts[llm_name] == CONSECUTIVE_FAILURES_TO_SKIP:
            logger.warning(f"⏭️  Skipping {llm_name} for remaining queries ({CONSECUTIVE_FAILURES_TO_SKIP} consecutive failures)")
    else:
        # Reset failure count on success
        llm_failure_counts[llm_name] = 0
    
    # Check if target is mentioned (case-insensitive check)
    target_mentioned = any(
        TARGET_COMPANY.lower() in company.lower() or company.lower() in TARGET_COMPANY.lower()
        for company in companies.keys()
    )
    
    return {
        "llm": llm_name,
        "intent": intent,
        "prompt": prompt,
        "run": run_num,
        "response": response,
        "companies_mentioned": companies,
        "target_mentioned": target_mentioned,
        "timestamp": datetime.now().isoformat()
    }

def run_audit():
    """Run the complete visibility audit."""
    print("\n" + "="*60)
    print("🔍 LLM VISIBILITY AUDIT FOR UPLERS")
    print("="*60)
    
    # Initialize clients
    available_clients = initialize_clients()
    if not available_clients:
        print("\n❌ No API clients available. Please set at least one environment variable:")
        print("   export OPENAI_API_KEY=your_key       # For ChatGPT")
        print("   export ANTHROPIC_API_KEY=your_key    # For Claude")
        print("   export GOOGLE_API_KEY=your_key       # For Gemini")
        print("   export XAI_API_KEY=your_key          # For Grok")
        print("   export PERPLEXITY_API_KEY=your_key   # For Perplexity")
        return None
    
    print(f"\n✅ Available LLMs: {', '.join(available_clients)}")
    
    # Determine which LLMs to query
    llms_to_query = []
    if "OpenAI" in available_clients:
        llms_to_query.append("ChatGPT")
    if "Anthropic" in available_clients:
        llms_to_query.append("Claude")
    if "Google" in available_clients:
        llms_to_query.append("Gemini")
    if "xAI" in available_clients:
        llms_to_query.append("Grok")
    if "Perplexity" in available_clients:
        llms_to_query.append("Perplexity")
    
    # Flatten prompts
    all_prompts = []
    for intent, prompts in PROMPTS_BY_INTENT.items():
        for prompt in prompts:
            all_prompts.append((intent, prompt))
    
    total_queries = len(all_prompts) * len(llms_to_query) * RUNS_PER_PROMPT
    print(f"📊 Running {total_queries} queries ({len(all_prompts)} prompts × {len(llms_to_query)} LLMs × {RUNS_PER_PROMPT} runs)")
    print(f"⏱️  Estimated time: {total_queries * 2 // 60} - {total_queries * 4 // 60} minutes\n")
    
    results = []
    completed = 0
    
    for intent, prompt in all_prompts:
        for llm in llms_to_query:
            for run_num in range(1, RUNS_PER_PROMPT + 1):
                try:
                    result = run_single_query(llm, prompt, intent, run_num)
                    results.append(result)
                    completed += 1
                    
                    # Progress indicator
                    target_status = "✓" if result["target_mentioned"] else "✗"
                    platforms_found = list(result["companies_mentioned"].keys())[:5]  # Show first 5
                    platforms_str = ", ".join(platforms_found) if platforms_found else "None"
                    print(f"[{completed}/{total_queries}] {llm} | {intent[:20]:20} | Target: {target_status} | Found: {platforms_str}")
                    
                    # Rate limiting
                    time.sleep(1)  # Adjust based on your rate limits
                    
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()
    
    return results

def analyze_results(results: list) -> dict:
    """Analyze results and compute metrics."""
    if not results:
        return {}
    
    analysis = {
        "meta": {
            "target_company": TARGET_COMPANY,
            "total_queries": len(results),
            "unique_prompts": len(set(r["prompt"] for r in results)),
            "llms_tested": list(set(r["llm"] for r in results)),
            "runs_per_prompt": RUNS_PER_PROMPT,
            "generated_at": datetime.now().isoformat()
        },
        "overall": {},
        "by_llm": {},
        "by_intent": {},
        "company_rankings": {},
        "weak_spots": []
    }
    
    # Overall metrics
    target_mentions = sum(1 for r in results if r["target_mentioned"])
    analysis["overall"]["visibility_score"] = round(target_mentions / len(results) * 100, 1)
    analysis["overall"]["target_mentions"] = target_mentions
    analysis["overall"]["total_queries"] = len(results)
    
    # By LLM
    for llm in analysis["meta"]["llms_tested"]:
        llm_results = [r for r in results if r["llm"] == llm]
        mentions = sum(1 for r in llm_results if r["target_mentioned"])
        analysis["by_llm"][llm] = {
            "visibility_score": round(mentions / len(llm_results) * 100, 1) if llm_results else 0,
            "mentions": mentions,
            "queries": len(llm_results)
        }
    
    # By Intent
    intents = set(r["intent"] for r in results)
    for intent in intents:
        intent_results = [r for r in results if r["intent"] == intent]
        mentions = sum(1 for r in intent_results if r["target_mentioned"])
        visibility = round(mentions / len(intent_results) * 100, 1) if intent_results else 0
        analysis["by_intent"][intent] = {
            "visibility_score": visibility,
            "mentions": mentions,
            "queries": len(intent_results)
        }
        
        # Identify weak spots (visibility < 20%)
        if visibility < 20:
            analysis["weak_spots"].append({
                "intent": intent,
                "visibility": visibility,
                "sample_prompts": list(set(r["prompt"] for r in intent_results))[:3]
            })
    
    # Company rankings (overall)
    company_counts = defaultdict(int)
    for result in results:
        for company, count in result["companies_mentioned"].items():
            company_counts[company] += count
    
    sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
    analysis["company_rankings"]["overall"] = [
        {"company": c, "mentions": m, "rank": i+1} 
        for i, (c, m) in enumerate(sorted_companies)
    ]
    
    # Find target's rank
    target_rank = next(
        (i+1 for i, (c, _) in enumerate(sorted_companies) if c == TARGET_COMPANY),
        len(sorted_companies) + 1
    )
    analysis["overall"]["target_rank"] = target_rank
    analysis["overall"]["total_companies_mentioned"] = len(sorted_companies)
    
    # Company rankings by LLM
    for llm in analysis["meta"]["llms_tested"]:
        llm_results = [r for r in results if r["llm"] == llm]
        llm_counts = defaultdict(int)
        for result in llm_results:
            for company, count in result["companies_mentioned"].items():
                llm_counts[company] += count
        
        sorted_llm = sorted(llm_counts.items(), key=lambda x: x[1], reverse=True)
        analysis["company_rankings"][llm] = [
            {"company": c, "mentions": m, "rank": i+1} 
            for i, (c, m) in enumerate(sorted_llm)
        ]
    
    return analysis

def generate_html_dashboard(analysis: dict, results: list) -> str:
    """Generate an interactive HTML dashboard."""
    
    # Get target rank for each LLM
    target_ranks = {}
    for llm in analysis["meta"]["llms_tested"]:
        rankings = analysis["company_rankings"].get(llm, [])
        rank = next((r["rank"] for r in rankings if r["company"] == TARGET_COMPANY), "N/A")
        target_ranks[llm] = rank
    
    # Prepare intent data for chart
    intent_data = []
    for intent, data in sorted(analysis["by_intent"].items(), key=lambda x: x[1]["visibility_score"], reverse=True):
        intent_data.append({
            "intent": intent,
            "score": data["visibility_score"],
            "mentions": data["mentions"],
            "queries": data["queries"]
        })
    
    # Prepare LLM comparison data
    llm_scores = {llm: data["visibility_score"] for llm, data in analysis["by_llm"].items()}
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Visibility Audit - {TARGET_COMPANY}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --accent: #00ff88;
            --accent-dim: #00cc6a;
            --text-primary: #ffffff;
            --text-secondary: #8b8b9e;
            --text-muted: #5a5a6e;
            --border: #2a2a3a;
            --danger: #ff4757;
            --warning: #ffa502;
            --success: #00ff88;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 24px;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 60px;
            position: relative;
        }}
        
        header::before {{
            content: '';
            position: absolute;
            top: -100px;
            left: 50%;
            transform: translateX(-50%);
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(0, 255, 136, 0.08) 0%, transparent 70%);
            pointer-events: none;
        }}
        
        .logo {{
            font-size: 14px;
            letter-spacing: 4px;
            color: var(--accent);
            text-transform: uppercase;
            margin-bottom: 16px;
        }}
        
        h1 {{
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #fff 0%, #00ff88 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 18px;
        }}
        
        .meta-info {{
            display: flex;
            justify-content: center;
            gap: 32px;
            margin-top: 24px;
            flex-wrap: wrap;
        }}
        
        .meta-item {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--text-muted);
        }}
        
        .meta-item span {{
            color: var(--accent);
        }}
        
        /* Score Cards */
        .score-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 48px;
        }}
        
        .score-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s, border-color 0.2s;
        }}
        
        .score-card:hover {{
            transform: translateY(-4px);
            border-color: var(--accent);
        }}
        
        .score-card.primary {{
            border-color: var(--accent);
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, var(--bg-card) 100%);
        }}
        
        .score-card .label {{
            font-size: 14px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }}
        
        .score-card .value {{
            font-size: 56px;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 8px;
        }}
        
        .score-card .value.accent {{ color: var(--accent); }}
        .score-card .value.warning {{ color: var(--warning); }}
        .score-card .value.danger {{ color: var(--danger); }}
        
        .score-card .detail {{
            font-size: 14px;
            color: var(--text-muted);
        }}
        
        /* Section Headers */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin: 48px 0 24px;
        }}
        
        .section-header h2 {{
            font-size: 24px;
            font-weight: 600;
        }}
        
        .section-header .line {{
            flex: 1;
            height: 1px;
            background: var(--border);
        }}
        
        /* LLM Comparison */
        .llm-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
            margin-bottom: 48px;
        }}
        
        .llm-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 28px;
        }}
        
        .llm-card .llm-name {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .llm-card .llm-icon {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
        }}
        
        .llm-icon.chatgpt {{ background: #10a37f; }}
        .llm-icon.claude {{ background: #d97757; }}
        .llm-icon.gemini {{ background: linear-gradient(135deg, #4285f4, #ea4335, #fbbc04, #34a853); }}
        .llm-icon.grok {{ background: #1a1a1a; border: 1px solid #333; }}
        .llm-icon.perplexity {{ background: #20b8cd; }}
        
        .llm-stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        
        .llm-stat {{
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 10px;
        }}
        
        .llm-stat .stat-label {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .llm-stat .stat-value {{
            font-size: 28px;
            font-weight: 600;
            margin-top: 4px;
        }}
        
        /* Rankings Table */
        .rankings-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 24px;
            margin-bottom: 48px;
        }}
        
        .rankings-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }}
        
        .rankings-card .card-header {{
            padding: 20px 24px;
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            font-size: 16px;
        }}
        
        .rankings-list {{
            max-height: 600px;
            overflow-y: auto;
        }}
        
        .ranking-item {{
            display: flex;
            align-items: center;
            padding: 14px 24px;
            border-bottom: 1px solid var(--border);
            transition: background 0.15s;
        }}
        
        .ranking-item:hover {{
            background: var(--bg-secondary);
        }}
        
        .ranking-item:last-child {{
            border-bottom: none;
        }}
        
        .ranking-item.target {{
            background: rgba(0, 255, 136, 0.1);
        }}
        
        .rank-num {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: var(--bg-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 14px;
            margin-right: 16px;
        }}
        
        .ranking-item.target .rank-num {{
            background: var(--accent);
            color: var(--bg-primary);
        }}
        
        .company-name {{
            flex: 1;
            font-weight: 500;
        }}
        
        .mention-count {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            font-size: 14px;
        }}
        
        /* Intent Analysis */
        .intent-chart-container {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 48px;
        }}
        
        .chart-wrapper {{
            height: 400px;
            position: relative;
        }}
        
        /* Weak Spots */
        .weak-spots {{
            background: linear-gradient(135deg, rgba(255, 71, 87, 0.1) 0%, var(--bg-card) 100%);
            border: 1px solid rgba(255, 71, 87, 0.3);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 48px;
        }}
        
        .weak-spots h3 {{
            color: var(--danger);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .weak-spot-item {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        
        .weak-spot-item:last-child {{
            margin-bottom: 0;
        }}
        
        .weak-spot-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        
        .weak-spot-intent {{
            font-weight: 600;
        }}
        
        .weak-spot-score {{
            color: var(--danger);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .weak-spot-prompts {{
            font-size: 14px;
            color: var(--text-muted);
        }}
        
        .weak-spot-prompts li {{
            margin: 8px 0;
            padding-left: 16px;
            position: relative;
        }}
        
        .weak-spot-prompts li::before {{
            content: '→';
            position: absolute;
            left: 0;
            color: var(--text-muted);
        }}
        
        /* Footer */
        footer {{
            text-align: center;
            padding: 40px 0;
            color: var(--text-muted);
            font-size: 14px;
            border-top: 1px solid var(--border);
            margin-top: 60px;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            h1 {{ font-size: 32px; }}
            .score-card .value {{ font-size: 40px; }}
            .container {{ padding: 24px 16px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">LLM Visibility Audit</div>
            <h1>{TARGET_COMPANY}</h1>
            <p class="subtitle">Brand visibility analysis across AI assistants</p>
            <div class="meta-info">
                <div class="meta-item">Generated: <span>{analysis["meta"]["generated_at"][:10]}</span></div>
                <div class="meta-item">Queries: <span>{analysis["meta"]["total_queries"]}</span></div>
                <div class="meta-item">Prompts: <span>{analysis["meta"]["unique_prompts"]}</span></div>
                <div class="meta-item">Runs/Prompt: <span>{analysis["meta"]["runs_per_prompt"]}</span></div>
            </div>
        </header>
        
        <!-- Overall Scores -->
        <div class="score-grid">
            <div class="score-card primary">
                <div class="label">Overall Visibility Score</div>
                <div class="value accent">{analysis["overall"]["visibility_score"]}%</div>
                <div class="detail">{analysis["overall"]["target_mentions"]} mentions across {analysis["overall"]["total_queries"]} queries</div>
            </div>
            <div class="score-card">
                <div class="label">Overall Ranking</div>
                <div class="value {'accent' if analysis["overall"]["target_rank"] <= 3 else 'warning' if analysis["overall"]["target_rank"] <= 10 else 'danger'}">#{analysis["overall"]["target_rank"]}</div>
                <div class="detail">out of {analysis["overall"]["total_companies_mentioned"]} companies mentioned</div>
            </div>
            <div class="score-card">
                <div class="label">Intent Categories</div>
                <div class="value">{len(analysis["by_intent"])}</div>
                <div class="detail">{len(analysis["weak_spots"])} weak spots identified</div>
            </div>
        </div>
        
        <!-- LLM Comparison -->
        <div class="section-header">
            <h2>Performance by LLM</h2>
            <div class="line"></div>
        </div>
        
        <div class="llm-grid">
'''
    
    # Add LLM cards
    for llm, data in analysis["by_llm"].items():
        # Map LLM names to CSS icon classes
        icon_class_map = {
            "ChatGPT": "chatgpt",
            "Claude": "claude",
            "Gemini": "gemini",
            "Grok": "grok",
            "Perplexity": "perplexity"
        }
        icon_class = icon_class_map.get(llm, llm.lower().replace(" ", ""))
        rank = target_ranks.get(llm, "N/A")
        html += f'''
            <div class="llm-card">
                <div class="llm-name">
                    <div class="llm-icon {icon_class}">{llm[0]}</div>
                    {llm}
                </div>
                <div class="llm-stats">
                    <div class="llm-stat">
                        <div class="stat-label">Visibility</div>
                        <div class="stat-value" style="color: var(--accent)">{data["visibility_score"]}%</div>
                    </div>
                    <div class="llm-stat">
                        <div class="stat-label">Rank</div>
                        <div class="stat-value">#{rank}</div>
                    </div>
                    <div class="llm-stat">
                        <div class="stat-label">Mentions</div>
                        <div class="stat-value">{data["mentions"]}</div>
                    </div>
                    <div class="llm-stat">
                        <div class="stat-label">Queries</div>
                        <div class="stat-value">{data["queries"]}</div>
                    </div>
                </div>
            </div>
'''
    
    html += '''
        </div>
        
        <!-- Intent Analysis Chart -->
        <div class="section-header">
            <h2>Visibility by Intent Category</h2>
            <div class="line"></div>
        </div>
        
        <div class="intent-chart-container">
            <div class="chart-wrapper">
                <canvas id="intentChart"></canvas>
            </div>
        </div>
'''
    
    # Weak Spots Section
    if analysis["weak_spots"]:
        html += '''
        <div class="weak-spots">
            <h3>⚠️ Weak Spots (Visibility < 20%)</h3>
'''
        for spot in analysis["weak_spots"]:
            html += f'''
            <div class="weak-spot-item">
                <div class="weak-spot-header">
                    <span class="weak-spot-intent">{spot["intent"]}</span>
                    <span class="weak-spot-score">{spot["visibility"]}%</span>
                </div>
                <ul class="weak-spot-prompts">
'''
            for prompt in spot["sample_prompts"]:
                html += f'                    <li>{prompt}</li>\n'
            html += '''
                </ul>
            </div>
'''
        html += '        </div>\n'
    
    # Rankings
    html += '''
        <div class="section-header">
            <h2>Company Rankings</h2>
            <div class="line"></div>
        </div>
        
        <div class="rankings-container">
            <div class="rankings-card">
                <div class="card-header">Overall Rankings</div>
                <div class="rankings-list">
'''
    
    for item in analysis["company_rankings"]["overall"][:30]:
        is_target = "target" if item["company"] == TARGET_COMPANY else ""
        html += f'''
                    <div class="ranking-item {is_target}">
                        <div class="rank-num">{item["rank"]}</div>
                        <span class="company-name">{item["company"]}</span>
                        <span class="mention-count">{item["mentions"]} mentions</span>
                    </div>
'''
    
    html += '''
                </div>
            </div>
'''
    
    # Add per-LLM rankings
    for llm in analysis["meta"]["llms_tested"]:
        if llm in analysis["company_rankings"]:
            html += f'''
            <div class="rankings-card">
                <div class="card-header">{llm} Rankings</div>
                <div class="rankings-list">
'''
            for item in analysis["company_rankings"][llm][:20]:
                is_target = "target" if item["company"] == TARGET_COMPANY else ""
                html += f'''
                    <div class="ranking-item {is_target}">
                        <div class="rank-num">{item["rank"]}</div>
                        <span class="company-name">{item["company"]}</span>
                        <span class="mention-count">{item["mentions"]} mentions</span>
                    </div>
'''
            html += '''
                </div>
            </div>
'''
    
    html += '''
        </div>
        
        <footer>
            <p>LLM Visibility Audit Tool • Built for strategic brand monitoring</p>
        </footer>
    </div>
    
    <script>
        // Intent Chart
        const ctx = document.getElementById('intentChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ''' + json.dumps([d["intent"] for d in intent_data]) + ''',
                datasets: [{
                    label: 'Visibility Score (%)',
                    data: ''' + json.dumps([d["score"] for d in intent_data]) + ''',
                    backgroundColor: function(context) {
                        const value = context.raw;
                        if (value >= 50) return '#00ff88';
                        if (value >= 20) return '#ffa502';
                        return '#ff4757';
                    },
                    borderRadius: 6,
                    borderSkipped: false,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1a1a24',
                        borderColor: '#2a2a3a',
                        borderWidth: 1,
                        titleFont: { family: 'Space Grotesk' },
                        bodyFont: { family: 'JetBrains Mono' },
                    }
                },
                scales: {
                    x: {
                        max: 100,
                        grid: { color: '#2a2a3a' },
                        ticks: { 
                            color: '#8b8b9e',
                            font: { family: 'JetBrains Mono' },
                            callback: function(value) { return value + '%'; }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { 
                            color: '#ffffff',
                            font: { family: 'Space Grotesk', size: 13 }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
'''
    
    return html


# ============================================================================
# SCREENSHOT CAPTURE
# ============================================================================

def capture_screenshot(html_file: str, output_png: str) -> bool:
    """Capture a screenshot of the HTML dashboard using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
        
        logger.info(f"📸 Capturing screenshot of {html_file}...")
        
        # Get absolute path
        html_path = os.path.abspath(html_file)
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1400, 'height': 900})
            
            # Load the HTML file
            page.goto(f"file://{html_path}")
            
            # Wait for chart to render
            page.wait_for_timeout(2000)
            
            # Capture full page screenshot
            page.screenshot(path=output_png, full_page=True)
            
            browser.close()
        
        logger.info(f"✅ Screenshot saved to {output_png}")
        return True
        
    except ImportError:
        logger.warning("⚠️  Playwright not installed. Run: pip install playwright && playwright install chromium")
        return False
    except Exception as e:
        logger.error(f"❌ Screenshot capture failed: {e}")
        return False


# ============================================================================
# HISTORICAL COMPARISON
# ============================================================================

def calculate_changes(current_analysis: dict, previous_data: dict) -> dict:
    """Calculate week-over-week changes in metrics."""
    changes = {
        "has_previous": False,
        "previous_date": None,
        "overall": {},
        "by_llm": {},
        "summary": []
    }
    
    if not previous_data or "analysis" not in previous_data:
        return changes
    
    prev_analysis = previous_data["analysis"]
    changes["has_previous"] = True
    changes["previous_date"] = previous_data.get("_archive_date", "Unknown")
    
    # Overall visibility change
    current_score = current_analysis["overall"]["visibility_score"]
    prev_score = prev_analysis.get("overall", {}).get("visibility_score", 0)
    score_change = round(current_score - prev_score, 1)
    
    changes["overall"]["visibility_score"] = {
        "current": current_score,
        "previous": prev_score,
        "change": score_change,
        "direction": "up" if score_change > 0 else "down" if score_change < 0 else "same"
    }
    
    # Rank change
    current_rank = current_analysis["overall"]["target_rank"]
    prev_rank = prev_analysis.get("overall", {}).get("target_rank", 999)
    rank_change = prev_rank - current_rank  # Positive = improved (lower rank is better)
    
    changes["overall"]["target_rank"] = {
        "current": current_rank,
        "previous": prev_rank,
        "change": rank_change,
        "direction": "up" if rank_change > 0 else "down" if rank_change < 0 else "same"
    }
    
    # By LLM changes
    for llm, data in current_analysis.get("by_llm", {}).items():
        prev_llm_data = prev_analysis.get("by_llm", {}).get(llm, {})
        prev_llm_score = prev_llm_data.get("visibility_score", 0)
        current_llm_score = data["visibility_score"]
        llm_change = round(current_llm_score - prev_llm_score, 1)
        
        changes["by_llm"][llm] = {
            "current": current_llm_score,
            "previous": prev_llm_score,
            "change": llm_change,
            "direction": "up" if llm_change > 0 else "down" if llm_change < 0 else "same"
        }
    
    # Generate summary
    if score_change > 0:
        changes["summary"].append(f"📈 Overall visibility improved by {score_change}%")
    elif score_change < 0:
        changes["summary"].append(f"📉 Overall visibility decreased by {abs(score_change)}%")
    else:
        changes["summary"].append("➡️ Overall visibility unchanged")
    
    if rank_change > 0:
        changes["summary"].append(f"🏆 Ranking improved by {rank_change} position(s)")
    elif rank_change < 0:
        changes["summary"].append(f"⬇️ Ranking dropped by {abs(rank_change)} position(s)")
    
    return changes


# ============================================================================
# EMAIL NOTIFICATIONS
# ============================================================================

def send_email_notification(analysis: dict, changes: dict, screenshot_path: str = None, dashboard_path: str = None) -> bool:
    """Send email notification with audit summary."""
    if not EMAIL_ENABLED:
        logger.info("📧 Email notifications disabled (no credentials configured)")
        return False
    
    try:
        logger.info(f"📧 Sending email notification to {EMAIL_TO}...")
        
        # Create message
        msg = MIMEMultipart('related')
        msg['Subject'] = f"🔍 LLM Visibility Audit Report - {TARGET_COMPANY} - {get_timestamp()}"
        msg['From'] = SMTP_USER
        msg['To'] = EMAIL_TO
        
        # Build HTML email body
        score = analysis["overall"]["visibility_score"]
        rank = analysis["overall"]["target_rank"]
        total_companies = analysis["overall"]["total_companies_mentioned"]
        
        # Determine score color
        if score >= 50:
            score_color = "#00ff88"
        elif score >= 20:
            score_color = "#ffa502"
        else:
            score_color = "#ff4757"
        
        # Build change indicators
        change_html = ""
        if changes.get("has_previous"):
            change_html = f"""
            <div style="background: #1a1a24; border-radius: 8px; padding: 16px; margin: 16px 0;">
                <h3 style="color: #8b8b9e; margin: 0 0 12px 0; font-size: 14px;">📊 WEEK-OVER-WEEK CHANGES</h3>
                <p style="color: #aaa; margin: 4px 0;">Compared to: {changes['previous_date']}</p>
            """
            for summary_item in changes.get("summary", []):
                change_html += f'<p style="color: #fff; margin: 8px 0;">{summary_item}</p>'
            change_html += "</div>"
        
        # Build LLM breakdown
        llm_rows = ""
        for llm, data in analysis.get("by_llm", {}).items():
            llm_change = changes.get("by_llm", {}).get(llm, {})
            change_indicator = ""
            if llm_change:
                if llm_change["direction"] == "up":
                    change_indicator = f'<span style="color: #00ff88;">↑{llm_change["change"]}%</span>'
                elif llm_change["direction"] == "down":
                    change_indicator = f'<span style="color: #ff4757;">↓{abs(llm_change["change"])}%</span>'
            
            llm_rows += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #2a2a3a; color: #fff;">{llm}</td>
                <td style="padding: 12px; border-bottom: 1px solid #2a2a3a; color: #00ff88;">{data['visibility_score']}%</td>
                <td style="padding: 12px; border-bottom: 1px solid #2a2a3a;">{data['mentions']}/{data['queries']}</td>
                <td style="padding: 12px; border-bottom: 1px solid #2a2a3a;">{change_indicator}</td>
            </tr>
            """
        
        # Weak spots
        weak_spots_html = ""
        if analysis.get("weak_spots"):
            weak_spots_html = """
            <div style="background: rgba(255, 71, 87, 0.1); border: 1px solid rgba(255, 71, 87, 0.3); border-radius: 8px; padding: 16px; margin: 16px 0;">
                <h3 style="color: #ff4757; margin: 0 0 12px 0;">⚠️ Weak Spots (Visibility < 20%)</h3>
                <ul style="color: #aaa; margin: 0; padding-left: 20px;">
            """
            for spot in analysis["weak_spots"]:
                weak_spots_html += f'<li style="margin: 8px 0;">{spot["intent"]}: {spot["visibility"]}%</li>'
            weak_spots_html += "</ul></div>"
        
        html_body = f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #fff; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto;">
                <div style="text-align: center; margin-bottom: 32px;">
                    <p style="color: #00ff88; font-size: 12px; letter-spacing: 2px; margin: 0;">LLM VISIBILITY AUDIT</p>
                    <h1 style="font-size: 28px; margin: 8px 0;">{TARGET_COMPANY}</h1>
                    <p style="color: #8b8b9e;">{get_timestamp()}</p>
                </div>
                
                <div style="display: flex; gap: 16px; margin-bottom: 24px;">
                    <div style="flex: 1; background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, #1a1a24 100%); border: 1px solid #00ff88; border-radius: 12px; padding: 24px; text-align: center;">
                        <p style="color: #8b8b9e; font-size: 12px; margin: 0;">VISIBILITY SCORE</p>
                        <p style="font-size: 48px; font-weight: bold; margin: 8px 0; color: {score_color};">{score}%</p>
                    </div>
                    <div style="flex: 1; background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 12px; padding: 24px; text-align: center;">
                        <p style="color: #8b8b9e; font-size: 12px; margin: 0;">RANKING</p>
                        <p style="font-size: 48px; font-weight: bold; margin: 8px 0; color: #fff;">#{rank}</p>
                        <p style="color: #5a5a6e; font-size: 12px; margin: 0;">of {total_companies} companies</p>
                    </div>
                </div>
                
                {change_html}
                
                <h3 style="color: #fff; margin: 24px 0 12px 0;">📊 Performance by LLM</h3>
                <table style="width: 100%; border-collapse: collapse; background: #1a1a24; border-radius: 8px; overflow: hidden;">
                    <thead>
                        <tr style="background: #12121a;">
                            <th style="padding: 12px; text-align: left; color: #8b8b9e; font-size: 12px;">LLM</th>
                            <th style="padding: 12px; text-align: left; color: #8b8b9e; font-size: 12px;">VISIBILITY</th>
                            <th style="padding: 12px; text-align: left; color: #8b8b9e; font-size: 12px;">MENTIONS</th>
                            <th style="padding: 12px; text-align: left; color: #8b8b9e; font-size: 12px;">CHANGE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {llm_rows}
                    </tbody>
                </table>
                
                {weak_spots_html}
                
                <div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #2a2a3a;">
                    <p style="color: #5a5a6e; font-size: 12px;">This is an automated report from the LLM Visibility Audit Tool.</p>
                    <p style="color: #5a5a6e; font-size: 12px;">Full dashboard attached.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Attach HTML body
        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)
        
        # Plain text version
        plain_text = f"""
LLM Visibility Audit Report - {TARGET_COMPANY}
Generated: {get_timestamp()}

OVERALL METRICS:
- Visibility Score: {score}%
- Ranking: #{rank} of {total_companies} companies

BY LLM:
"""
        for llm, data in analysis.get("by_llm", {}).items():
            plain_text += f"- {llm}: {data['visibility_score']}% ({data['mentions']}/{data['queries']} queries)\n"
        
        if analysis.get("weak_spots"):
            plain_text += "\nWEAK SPOTS:\n"
            for spot in analysis["weak_spots"]:
                plain_text += f"- {spot['intent']}: {spot['visibility']}%\n"
        
        msg_alternative.attach(MIMEText(plain_text, 'plain'))
        msg_alternative.attach(MIMEText(html_body, 'html'))
        
        # Attach screenshot if available
        if screenshot_path and os.path.exists(screenshot_path):
            with open(screenshot_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(screenshot_path))
                msg.attach(img)
        
        # Attach dashboard HTML if available
        if dashboard_path and os.path.exists(dashboard_path):
            with open(dashboard_path, 'rb') as f:
                part = MIMEBase('text', 'html')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(dashboard_path))
                msg.attach(part)
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"✅ Email sent successfully to {EMAIL_TO}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")
        traceback.print_exc()
        return False


def test_email():
    """Send a test email to verify configuration."""
    if not EMAIL_ENABLED:
        print("❌ Email not configured. Set these environment variables:")
        print("   SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_TO")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['Subject'] = f"🔍 LLM Visibility Audit - Test Email"
        msg['From'] = SMTP_USER
        msg['To'] = EMAIL_TO
        
        body = f"""
        This is a test email from the LLM Visibility Audit Tool.
        
        Configuration:
        - SMTP Host: {SMTP_HOST}
        - SMTP Port: {SMTP_PORT}
        - From: {SMTP_USER}
        - To: {EMAIL_TO}
        
        If you received this, your email configuration is working correctly!
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        print(f"✅ Test email sent successfully to {EMAIL_TO}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send test email: {e}")
        traceback.print_exc()
        return False


def show_cron_setup():
    """Display cron setup instructions."""
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    
    print("\n" + "="*60)
    print("📅 CRON SETUP INSTRUCTIONS")
    print("="*60)
    print("""
To run this script automatically every week, add a cron job:

1. Open your crontab:
   crontab -e

2. Add one of these lines (choose your preferred schedule):

   # Run every Sunday at 6:00 AM
   0 6 * * 0 cd {dir} && {python} {script} >> {dir}/cron.log 2>&1

   # Run every Monday at 9:00 AM
   0 9 * * 1 cd {dir} && {python} {script} >> {dir}/cron.log 2>&1

   # Run every Saturday at midnight
   0 0 * * 6 cd {dir} && {python} {script} >> {dir}/cron.log 2>&1

3. Make sure your .env file is in the same directory with your API keys.

4. (Optional) Set up a systemd service for more robust scheduling:

   Create /etc/systemd/system/visibility-audit.service:
   ---
   [Unit]
   Description=LLM Visibility Audit
   After=network.target

   [Service]
   Type=oneshot
   WorkingDirectory={dir}
   ExecStart={python} {script}
   User={user}
   Environment=PATH=/usr/local/bin:/usr/bin:/bin

   [Install]
   WantedBy=multi-user.target
   ---

   Create /etc/systemd/system/visibility-audit.timer:
   ---
   [Unit]
   Description=Run LLM Visibility Audit Weekly

   [Timer]
   OnCalendar=Sun 06:00
   Persistent=true

   [Install]
   WantedBy=timers.target
   ---

   Then enable: sudo systemctl enable visibility-audit.timer
""".format(
        dir=os.path.dirname(script_path),
        python=python_path,
        script=script_path,
        user=os.getenv('USER', 'your_user')
    ))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Visibility Audit Tool')
    parser.add_argument('--test-email', action='store_true', help='Send a test email')
    parser.add_argument('--setup-cron', action='store_true', help='Show cron setup instructions')
    parser.add_argument('--no-email', action='store_true', help='Skip sending email notification')
    parser.add_argument('--no-screenshot', action='store_true', help='Skip screenshot capture')
    args = parser.parse_args()
    
    # Handle special commands
    if args.test_email:
        test_email()
        return
    
    if args.setup_cron:
        show_cron_setup()
        return
    
    # Start audit
    logger.info("\n" + "="*60)
    logger.info("🚀 STARTING LLM VISIBILITY AUDIT")
    logger.info("="*60)
    logger.info(f"Target Company: {TARGET_COMPANY}")
    logger.info(f"Target Region: {TARGET_REGION}")
    logger.info(f"Prompt Categories: {len(PROMPTS_BY_INTENT)}")
    logger.info(f"Total Prompts: {sum(len(p) for p in PROMPTS_BY_INTENT.values())}")
    logger.info(f"Runs per Prompt: {RUNS_PER_PROMPT}")
    logger.info(f"Archive Retention: {ARCHIVE_RETENTION_WEEKS} weeks")
    logger.info(f"Email Notifications: {'Enabled' if EMAIL_ENABLED else 'Disabled'}")
    
    # Get previous results for comparison
    previous_data = get_previous_results()
    if previous_data:
        logger.info(f"📊 Found previous audit from {previous_data.get('_archive_date', 'unknown date')}")
    
    # Run audit
    try:
        results = run_audit()
    except Exception as e:
        logger.error(f"❌ Audit failed with error: {e}")
        traceback.print_exc()
        return
    
    if not results:
        logger.error("❌ No results collected. Exiting.")
        return
    
    # Analyze
    logger.info("📊 Analyzing results...")
    analysis = analyze_results(results)
    
    # Calculate changes from previous run
    changes = calculate_changes(analysis, previous_data)
    
    # Create timestamp for this run
    timestamp = get_timestamp()
    ensure_archive_dir()
    
    # Save current results
    results_file = "audit_results.json"
    archive_results_file = f"{ARCHIVE_DIR}/audit_results_{timestamp}.json"
    
    results_data = {"analysis": analysis, "raw_results": results}
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    shutil.copy(results_file, archive_results_file)
    logger.info(f"✅ Results saved to {results_file}")
    logger.info(f"📁 Archived to {archive_results_file}")
    
    # Generate dashboard
    logger.info("🎨 Generating HTML dashboard...")
    dashboard_html = generate_html_dashboard(analysis, results)
    
    dashboard_file = "visibility_dashboard.html"
    archive_dashboard_file = f"{ARCHIVE_DIR}/visibility_dashboard_{timestamp}.html"
    
    with open(dashboard_file, "w") as f:
        f.write(dashboard_html)
    shutil.copy(dashboard_file, archive_dashboard_file)
    logger.info(f"✅ Dashboard saved to {dashboard_file}")
    logger.info(f"📁 Archived to {archive_dashboard_file}")
    
    # Capture screenshot
    screenshot_file = None
    if not args.no_screenshot:
        screenshot_file = f"visibility_dashboard_{timestamp}.png"
        archive_screenshot_file = f"{ARCHIVE_DIR}/visibility_dashboard_{timestamp}.png"
        
        if capture_screenshot(dashboard_file, screenshot_file):
            shutil.copy(screenshot_file, archive_screenshot_file)
            logger.info(f"📁 Screenshot archived to {archive_screenshot_file}")
    
    # Clean up old archives
    cleanup_old_archives()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📈 AUDIT SUMMARY")
    logger.info("="*60)
    logger.info(f"🎯 Overall Visibility Score: {analysis['overall']['visibility_score']}%")
    logger.info(f"🏆 Overall Ranking: #{analysis['overall']['target_rank']} of {analysis['overall']['total_companies_mentioned']}")
    
    # Show changes if available
    if changes.get("has_previous"):
        logger.info(f"\n📊 Changes since {changes['previous_date']}:")
        for summary in changes.get("summary", []):
            logger.info(f"   {summary}")
    
    logger.info("\n📊 By LLM:")
    for llm, data in analysis["by_llm"].items():
        change_str = ""
        if changes.get("by_llm", {}).get(llm):
            llm_change = changes["by_llm"][llm]
            if llm_change["direction"] == "up":
                change_str = f" (↑{llm_change['change']}%)"
            elif llm_change["direction"] == "down":
                change_str = f" (↓{abs(llm_change['change'])}%)"
        logger.info(f"   {llm}: {data['visibility_score']}%{change_str} ({data['mentions']}/{data['queries']} queries)")
    
    if analysis["weak_spots"]:
        logger.info("\n⚠️  Weak Spots:")
        for spot in analysis["weak_spots"]:
            logger.info(f"   - {spot['intent']}: {spot['visibility']}%")
    
    # Send email notification
    if not args.no_email and EMAIL_ENABLED:
        send_email_notification(analysis, changes, screenshot_file, dashboard_file)
    
    logger.info("\n✅ Audit complete!")
    logger.info(f"   Dashboard: {dashboard_file}")
    logger.info(f"   Raw data: {results_file}")
    if screenshot_file:
        logger.info(f"   Screenshot: {screenshot_file}")
    logger.info(f"   Archives: {ARCHIVE_DIR}/")


if __name__ == "__main__":
    main()