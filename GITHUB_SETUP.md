# 🚀 GitHub Actions + GitHub Pages Setup Guide

This guide will help you deploy your LLM Visibility Dashboard automatically using GitHub Actions.

---

## 📋 What You'll Get

- ✅ **Automated weekly runs** - Dashboard updates every Sunday at 6 AM UTC
- ✅ **Free hosting** - GitHub Pages serves your dashboard
- ✅ **Public URL** - `https://YOUR_USERNAME.github.io/REPO_NAME/`
- ✅ **Manual trigger** - Run anytime from GitHub UI
- ✅ **Version history** - All updates tracked in git

---

## 🛠️ Step-by-Step Setup

### Step 1: Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Create a new repository:
   - **Name**: `llm-visibility-audit` (or any name you prefer)
   - **Visibility**: Public (required for free GitHub Pages)
   - **Don't** initialize with README (we'll push our files)

### Step 2: Push Your Code to GitHub

```bash
cd /Users/up2721/Desktop/test-script

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: LLM Visibility Audit Tool"

# Add your GitHub repo as remote
git remote add origin https://github.com/YOUR_USERNAME/llm-visibility-audit.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Add API Keys as Secrets

1. Go to your repo on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add each:

| Secret Name | Value | Required |
|-------------|-------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | At least one |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | At least one |
| `GOOGLE_API_KEY` | Your Google Gemini API key | Optional |
| `XAI_API_KEY` | Your xAI Grok API key | Optional |
| `PERPLEXITY_API_KEY` | Your Perplexity API key | Optional |
| `SMTP_HOST` | `smtp.gmail.com` | For email |
| `SMTP_PORT` | `587` | For email |
| `SMTP_USER` | Your email | For email |
| `SMTP_PASSWORD` | Your app password | For email |
| `EMAIL_TO` | Recipient email | For email |

### Step 4: Enable GitHub Pages

1. Go to your repo **Settings** → **Pages**
2. Under **Source**, select **GitHub Actions**
3. Save

### Step 5: Run the Workflow (First Time)

1. Go to **Actions** tab in your repo
2. Click **LLM Visibility Audit** workflow
3. Click **Run workflow** → **Run workflow**
4. Wait for it to complete (~30-60 minutes)

### Step 6: Access Your Dashboard

Once the workflow completes, your dashboard will be live at:

```
https://YOUR_USERNAME.github.io/llm-visibility-audit/
```

---

## 📅 Automatic Schedule

The workflow runs automatically every **Sunday at 6:00 AM UTC**.

To change the schedule, edit `.github/workflows/visibility-audit.yml`:

```yaml
schedule:
  # Run every Sunday at 6:00 AM UTC
  - cron: '0 6 * * 0'
  
  # Other examples:
  # Every Monday at 9 AM UTC: '0 9 * * 1'
  # Every day at midnight: '0 0 * * *'
  # Every Saturday at 3 PM UTC: '0 15 * * 6'
```

---

## 🔧 Manual Run

You can trigger the workflow manually anytime:

1. Go to **Actions** tab
2. Click **LLM Visibility Audit**
3. Click **Run workflow**

---

## 📁 Repository Structure

After running, your repo will look like:

```
llm-visibility-audit/
├── .github/
│   └── workflows/
│       └── visibility-audit.yml    # GitHub Actions workflow
├── docs/                           # GitHub Pages content
│   ├── index.html                  # Dashboard (served as webpage)
│   ├── audit_results.json          # Raw data
│   └── archives/                   # Historical data
├── visibility_audit2.0.py          # Main script
├── requirements.txt                # Dependencies
├── audit_results.json              # Latest results
├── visibility_dashboard.html       # Latest dashboard
└── archives/                       # Local archives
```

---

## ⚠️ Important Notes

1. **Public Repository Required**: Free GitHub Pages requires a public repo. If you need private, you'll need GitHub Pro/Team.

2. **Workflow Minutes**: GitHub gives 2,000 free minutes/month for public repos. Each run takes ~30-60 minutes.

3. **API Costs**: Running the audit uses API credits from each LLM provider. Budget accordingly.

4. **Secrets Security**: Never commit API keys to code. Always use GitHub Secrets.

---

## 🔗 Custom Domain (Optional)

To use a custom domain like `visibility.mavlers.com`:

1. Go to repo **Settings** → **Pages**
2. Under **Custom domain**, enter `visibility.mavlers.com`
3. Add a CNAME record in your DNS:
   ```
   visibility.mavlers.com → YOUR_USERNAME.github.io
   ```

---

## 🐛 Troubleshooting

### Workflow Fails
- Check **Actions** tab for error logs
- Ensure all required secrets are set
- Verify API keys are valid

### Dashboard Not Updating
- Check if workflow ran successfully
- Clear browser cache
- Wait a few minutes for Pages to deploy

### Rate Limits
- Reduce number of prompts
- Increase delays between requests
- Use paid API tiers

---

## ✅ You're All Set!

Your LLM Visibility Dashboard will now:
1. 🔄 Run automatically every Sunday
2. 📊 Generate fresh visibility scores
3. 🌐 Deploy to GitHub Pages
4. 📧 Send email notification (if configured)

Dashboard URL: `https://YOUR_USERNAME.github.io/llm-visibility-audit/`

