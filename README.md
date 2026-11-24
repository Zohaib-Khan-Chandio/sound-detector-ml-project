# Environmental Sound Detector - Streamlit version

Same page design as your website, same trained model. Runs on Streamlit Community Cloud (free, no card).

## Files in this folder
| File | Where it comes from |
|---|---|
| streamlit_app.py, core.py, frontend/index.html, requirements.txt, packages.txt, .streamlit/config.toml | this zip |
| features_v2.py | this zip (it is the same file Part 2B saved) |
| **model.pkl, classes.json, config.json** | **from your Drive: EnvSoundDetector/website/** (made by Part 4) |

## Test on your laptop first
    pip install -r requirements.txt
    streamlit run streamlit_app.py

## Deploy (free)
1. Create a free GitHub account, then a new PUBLIC repository.
2. Upload ALL files above (keep the frontend and .streamlit folders). model.pkl over 25 MB: use GitHub Desktop or git instead of the web upload.
3. Go to share.streamlit.io, sign in with GitHub, click "Create app".
4. Pick your repository, branch main, main file path: streamlit_app.py
5. Open "Advanced settings" and choose Python 3.12 (or 3.11).
6. Click Deploy and wait a few minutes. You get a public https link (the microphone needs https).
