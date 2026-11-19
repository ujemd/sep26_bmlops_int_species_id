"""Streamlit front end: submit a photo, see the identification, correct it."""
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Species ID", page_icon="🦋", layout="wide")
st.title("Species identification")

try:
    info = requests.get(f"{API}/model", timeout=5).json()
    st.caption(f"Model v{info['version']} · {info['extractor']} · {len(info['classes'])} taxa · "
               f"dataset {str(info.get('dataset_version'))[:8]}")
    classes = info["classes"]
except Exception:
    st.warning("The API has no model yet, or is unreachable.")
    classes = []

upload = st.file_uploader("Photo of an organism", type=["jpg", "jpeg", "png", "webp"])
if upload:
    left, right = st.columns([1, 1])
    left.image(upload, use_container_width=True)
    r = requests.post(f"{API}/predict", files={"file": (upload.name, upload.getvalue())}, timeout=60)
    if r.status_code != 200:
        right.error(r.json().get("detail", r.text))
        st.stop()
    out = r.json()
    if out.get("out_of_scope"):
        right.error(f"This photo does not look like any of the {len(classes)} species the model knows "
                    f"(closest guess: {out['taxon']}). Sent to human review.")
    elif out["needs_human_review"]:
        right.warning(f"Uncertain: **{out['taxon']}** ({out['confidence']:.0%}). "
                      f"Sent to human review. Plausible species: {', '.join(out['prediction_set']) or 'none'}")
    else:
        right.success(f"**{out['taxon']}** · {out['confidence']:.0%} confidence")
    right.bar_chart(pd.DataFrame(out["top_k"]).set_index("taxon"), horizontal=True)
    right.caption(f"model v{out['model_version']} · {out['latency_ms']:.0f} ms · prediction #{out['prediction_id']}")

    with right.form("feedback"):
        truth = st.selectbox("Correct species", classes, index=classes.index(out["taxon"]) if classes else None)
        if st.form_submit_button("Send correction"):
            requests.post(f"{API}/feedback", json={"prediction_id": out["prediction_id"], "true_taxon": truth},
                          timeout=10)
            st.success("Thanks, recorded for the next retraining.")
