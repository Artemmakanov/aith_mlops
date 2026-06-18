import os
import time

import requests
import streamlit as st

DEFAULT_ENDPOINT = "http://localhost:9008/serve/sentiment"
ENDPOINT_URL = os.getenv("SERVING_URL", DEFAULT_ENDPOINT)

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title("Sentiment Analysis")
st.caption("Predictions via HTTP — model is served remotely, not loaded in this UI.")

with st.sidebar:
    endpoint = st.text_input("Endpoint URL", value=ENDPOINT_URL)
    st.caption("Override with `SERVING_URL` env var.")

text = st.text_area(
    "Text",
    placeholder="Enter a sentence to classify...",
    height=120,
)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            start = time.perf_counter()
            response = requests.post(
                endpoint,
                json={"x": [text.strip()]},
                timeout=30,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                predictions = response.json()
                label = predictions[0] if predictions else "—"
                st.success(f"**Label:** {label}")
                st.metric("Latency", f"{latency_ms:.0f} ms")
            else:
                st.error(
                    f"Server returned {response.status_code}: {response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Endpoint is unavailable. "
                "Check that ClearML Serving is running and the URL is correct."
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server did not respond in time.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
