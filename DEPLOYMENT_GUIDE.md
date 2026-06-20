# 🚀 Deploy Crude Oil Price Predictor to Streamlit Cloud

## What You'll Get
- **Free hosting** (Streamlit Community Cloud)
- **Permanent public URL** (shareable link)
- **Automatic updates** (redeploy by pushing to GitHub)
- **Live WTI price data** (updates automatically)

---

## Step 1: Create a GitHub Repository

1. Go to **https://github.com/new**
2. Create a new repo called `oil-price-predictor`
3. Set it to **Public** (required for free Streamlit Cloud)
4. Click **Create repository**

---

## Step 2: Add Files to Your Repo

You need 3 files in the repo root:

### File 1: `app.py`
- Copy the full `oil_predictor_app.py` code
- Rename it to `app.py` and upload to your repo

### File 2: `requirements.txt`
```
streamlit>=1.28.0
yfinance>=0.2.32
scikit-learn>=1.3.0
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
```

### File 3: `.streamlit/config.toml` (optional, for styling)
```toml
[theme]
primaryColor = "#f0a500"
backgroundColor = "#0f1117"
secondaryBackgroundColor = "#1c1f2e"
textColor = "#e0e0e0"
font = "sans serif"

[client]
showErrorDetails = false
```

Create a folder `.streamlit/` in your repo and put `config.toml` inside.

---

## Step 3: Connect to Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Click **"New app"** (top-left)
3. Sign in with your GitHub account (if prompted)
4. Fill in:
   - **Repository**: `username/oil-price-predictor`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy**

Streamlit will build and launch your app. This takes 1–2 minutes.

---

## Step 4: Share Your App

Once deployed, you'll get a URL like:
```
https://oil-price-predictor-xxxxxx.streamlit.app
```

Share this link with anyone. They can use the app without installing anything.

---

## How to Update Your App

Just push changes to GitHub:
```bash
git add .
git commit -m "Update impact factors"
git push origin main
```

Streamlit Cloud automatically redeploys within 1–2 minutes.

---

## Troubleshooting

**"App failed to build"**
- Check `requirements.txt` for typos
- Make sure `app.py` is in the repo root (not in a subfolder)
- Check the **Logs** tab in Streamlit Cloud dashboard

**"yfinance failed to download"**
- Yahoo Finance can have brief outages
- The app caches data for 1 hour, so refresh after waiting

**"App is slow"**
- First load takes ~30 seconds (installing dependencies)
- Subsequent loads are instant (models are cached)

---

## File Structure

Your GitHub repo should look like:
```
oil-price-predictor/
├── app.py
├── requirements.txt
├── README.md (optional)
└── .streamlit/
    └── config.toml
```

---

## Done!

Your app is now live at a permanent, shareable URL. Anyone can use it without installing anything.
