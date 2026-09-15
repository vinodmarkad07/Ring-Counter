# Ring Stack Counter — Web App (Full Guide)

This is a small Flask web app that does exactly what you asked for:
- Deploy once → get a permanent URL.
- Open that URL on any phone or laptop browser.
- Tap "Take / Choose Photo" → phone opens its **native camera** (no app
  install needed, it's just a browser file-input with `capture="environment"`).
- Photo uploads to your server → server runs the ring-counting algorithm →
  you see the count + annotated image right in the browser.

No Pydroid, no Kivy, no phone-side Python at all. The phone only needs a
browser. All the actual processing (OpenCV, scipy) runs on the server.

## Project structure
```
ring-web/
├── app.py                  # Flask server + ring-counting algorithm
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Camera-capture page (mobile friendly)
├── uploads/                # incoming photos (auto-created)
└── outputs/                # annotated result images (auto-created)
```

## 1. Run it locally first (sanity check)
```bash
cd ring-web
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000` on your computer. On your **phone**, connect
to the same WiFi and open `http://<your-computer's-LAN-IP>:5000` — you'll
already get a working camera-to-count flow before deploying anywhere.

## 2. Deploy for a permanent public URL — Render.com (recommended, free)

Render is the easiest free option for a Flask + OpenCV app (no credit
card, handles Python builds natively).

**Steps:**
1. Push this `ring-web/` folder to a GitHub repo.
2. Go to https://render.com → sign up (GitHub login is fastest).
3. Click **New +** → **Web Service** → connect your GitHub repo.
4. Fill in:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click **Create Web Service**.
6. Wait ~2–3 minutes for the build. Render gives you a URL like:
   `https://ring-counter.onrender.com`
7. Open that URL on your phone. Done — that's your permanent app link.
   Bookmark it or "Add to Home Screen" so it opens like an app icon.

Free tier note: the service sleeps after 15 min of no traffic and takes
~30–50 seconds to wake up on the next request. Fine for occasional use;
upgrade to a paid instance ($7/mo) if you need it always-on/instant.

## 3. Alternative deploy targets (same code, no changes needed)
- **Railway.app** — similar to Render, also has a free trial tier.
- **PythonAnywhere** — good if you want a very simple always-on free
  Flask host (slightly more manual WSGI setup).
- **Your own VPS / server at Technohertz** — run with:
  ```bash
  gunicorn -w 2 -b 0.0.0.0:8000 app:app
  ```
  then put Nginx in front for HTTPS, or use a service like Caddy for
  automatic HTTPS with zero config.

## 4. "Or create an app and use it" — turning this into an installable app
You don't need a separate native app. Once deployed:
- On Android Chrome: open the URL → menu → **"Add to Home screen"**.
- On iPhone Safari: Share button → **"Add to Home Screen"**.

This is a **PWA-style shortcut** — it gets an icon on the home screen and
opens full-screen like a real app, but it's still just your web page
under the hood. No app store, no APK, and any algorithm update you push
to the server is instantly live for everyone using it.

## How the algorithm works (same as your desktop counter.py)
1. Resize photo, convert to grayscale.
2. CLAHE contrast enhancement (makes faint ring seams visible).
3. Auto-detect the stack region (Otsu threshold + largest contour) —
   works automatically server-side, no manual tap/ROI needed since the
   browser just sends a plain photo.
4. Extract a vertical intensity profile through the center of the stack.
5. Local contrast normalization + gradient → sharp spikes at ring
   boundaries.
6. `scipy.signal.find_peaks` finds those spikes; gap-consistency
   filtering drops spurious ones.
7. Ring count = peaks − 1. Confidence (High/Medium/Low) factors in blur,
   brightness, glare, and how consistent the ring spacing was.

## If counts come out wrong
- The current test run on your sample images gave ring_count with
  "Medium" confidence and `box_reliable: false` — meaning Otsu auto-detect
  didn't cleanly isolate the stack on a busy background. Two ways to fix:
  1. Ask users to photograph the stack against a plain/contrasting
     background, filling most of the frame.
  2. Bring back manual ROI: add a second step to `index.html` where the
     user taps top/bottom on the uploaded photo (like your original
     "Ring Stack Gauge" prototype) and send those coordinates to `/count`
     instead of relying on auto-detect. I can add this if auto-detect
     proves unreliable in practice — it's a small change to both the
     HTML (canvas tap capture) and `app.py` (accept `top_y`/`bottom_y`
     params instead of calling `detect_stack_auto`).
