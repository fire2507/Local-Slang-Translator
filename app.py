"""Local Slang Translator - Consolidated Streamlit Application."""

import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import easyocr
from gtts import gTTS
from groq import Groq
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import speech_recognition as sr
import streamlit as st

# Load environment configuration
load_dotenv()

# ==============================================================================
# 1. CONFIGURATIONS & DICTIONARIES
# ==============================================================================

LANGUAGES: Dict[str, Dict[str, str]] = {
    "English": {
        "code": "en",
        "gtts": "en",
        "speech_rec": "en-US",
        "easyocr": "en",
    },
    "Hindi": {
        "code": "hi",
        "gtts": "hi",
        "speech_rec": "hi-IN",
        "easyocr": "hi",
    },
    "Korean": {
        "code": "ko",
        "gtts": "ko",
        "speech_rec": "ko-KR",
        "easyocr": "ko",
    },
    "Spanish": {
        "code": "es",
        "gtts": "es",
        "speech_rec": "es-ES",
        "easyocr": "es",
    },
}

OFFLINE_SLANG_DICTIONARY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "English": {
        "ghosted": {
            "professional": "stopped responding without notice",
            "meaning": "Suddenly cut off all communication",
        },
        "no cap": {
            "professional": "without exaggeration",
            "meaning": "To speak honestly without lying or exaggerating",
        },
        "goat": {
            "professional": "greatest of all time",
            "meaning": "Acronym for Greatest Of All Time",
        },
        "salty": {
            "professional": "upset or bitter",
            "meaning": "Being upset or agitated over something minor",
        },
    },
    "Hindi": {
        "jugaad": {
            "professional": "creative workaround",
            "meaning": "An innovative fix or flexible solution",
        },
        "faff": {
            "professional": "unnecessary talking",
            "meaning": "Waste time in useless activity or talking",
        },
        "bindaas": {
            "professional": "carefree and confident",
            "meaning": "Relaxed and unconcerned about consequences",
        },
    },
    "Korean": {
        "대박": {
            "professional": "대단히 놀랍습니다 (very impressive)",
            "meaning": "Awesome, amazing, or big success",
        },
        "멘붕": {
            "professional": "정신적 혼란 (mental confusion)",
            "meaning": "Mental breakdown or extreme panic",
        },
        "꿀잼": {
            "professional": "매우 흥미롭습니다 (very interesting)",
            "meaning": "Super fun or exciting",
        },
    },
    "Spanish": {
        "chamba": {
            "professional": "trabajo (job/work)",
            "meaning": "Work or employment",
        },
        "guay": {
            "professional": "excelente (excellent)",
            "meaning": "Cool, awesome, or great",
        },
        "onda": {
            "professional": "situación o actitud (vibe/situation)",
            "meaning": "Good vibe or friendly attitude",
        },
    },
}


def get_lang_config(language: str) -> Dict[str, str]:
    """Retrieves service language codes for a given language name."""
    return LANGUAGES.get(language, LANGUAGES["English"])


# ==============================================================================
# 2. SERVICES (TRANSLATION, SPEECH, OCR)
# ==============================================================================

class TranslationError(Exception):
    """Custom exception raised during translation errors."""
    pass


def get_groq_client() -> Groq:
    """Initializes and returns Groq client using environment variable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise TranslationError("GROQ_API_KEY is missing from environment.")
    return Groq(api_key=api_key)


def build_system_prompt(language: str, mode: str, tone: str) -> str:
    """Constructs prompt forcing strict JSON format for translation."""
    return f"""You are an expert linguistic translator specializing in regional slang, idioms, and professional communication in {language}.

TASK:
Convert the provided input text.
- If Mode is 'Slang → Professional': Translate informal/slang/regional terms into modern standard {language} with a {tone} tone.
- If Mode is 'Professional → Slang': Convert formal/professional {language} into natural, authentic regional slang in {language} with a {tone} tone.

CRITICAL INSTRUCTIONS:
1. Preserve the core meaning, intent, names, numbers, and dates accurately.
2. Return ONLY a valid JSON object. Do not wrap in markdown ```json or include preambles/postscript.
3. The JSON structure MUST follow this exact schema:
{{
  "translation": "translated text here",
  "detected_slang": [
    {{
      "term": "slang or idiom identified",
      "meaning": "clear explanation of what it means in context"
    }}
  ]
}}
4. If no slang terms are present or when converting Professional → Slang, provide relevant key slang terms introduced in the output under "detected_slang".
"""


def parse_llm_json(raw_response: str) -> Dict[str, Any]:
    """Extracts and safely parses JSON content from LLM response text."""
    cleaned = raw_response.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
        if "translation" in data and "detected_slang" in data:
            return data
        raise ValueError("JSON response missing required keys.")
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON output: {str(err)}")


def offline_fallback_translate(text: str, language: str, mode: str) -> Dict[str, Any]:
    """Basic offline fallback translation when API is unavailable."""
    dict_lang = OFFLINE_SLANG_DICTIONARY.get(language, {})
    detected: List[Dict[str, str]] = []
    translated_text = text

    text_lower = text.lower()
    for slang_term, info in dict_lang.items():
        if slang_term in text_lower:
            detected.append({"term": slang_term, "meaning": info["meaning"]})
            if mode == "Slang → Professional":
                pattern = re.compile(re.escape(slang_term), re.IGNORECASE)
                translated_text = pattern.sub(info["professional"], translated_text)

    if not detected:
        translated_text = f"[Offline Fallback] {text}"

    return {
        "translation": translated_text,
        "detected_slang": detected,
        "offline": True,
    }


def translate_text(text: str, language: str, mode: str, tone: str) -> Dict[str, Any]:
    """Translates text using Groq LLM API with dynamic active model discovery."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    system_prompt = build_system_prompt(language, mode, tone)

    try:
        client = get_groq_client()
        
        # 1. Fetch available models directly from Groq API for your key
        available_models = [m.id for m in client.models.list().data]
        
        # 2. Preferred order of standard text generation models
        preferred_order = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
        ]
        
        # Build prioritized list based on what is actually available on your account
        models_to_try = [m for m in preferred_order if m in available_models]
        
        # If none of the preferred match, fall back to any active non-whisper/non-guard model
        if not models_to_try:
            models_to_try = [
                m for m in available_models 
                if not any(x in m for x in ["whisper", "guard", "vision"])
            ]

        if not models_to_try:
            raise TranslationError("No compatible text generation models available for this API key.")

        last_exception = None

        # 3. Iterate through available models
        for model_name in models_to_try:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Text to convert: {text.strip()}"},
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                )

                response_content = completion.choices[0].message.content
                result = parse_llm_json(response_content)
                result["offline"] = False
                return result

            except Exception as e:
                last_exception = e
                continue

        raise last_exception or TranslationError("All available Groq model attempts failed.")

    except Exception as e:
        fallback_result = offline_fallback_translate(text, language, mode)
        fallback_result["warning"] = f"API Error ({str(e)}). Using offline fallback."
        return fallback_result


def transcribe_audio(audio_bytes: bytes, language: str) -> str:
    """Transcribes audio byte stream using SpeechRecognition."""
    if not audio_bytes:
        raise ValueError("No audio data provided.")

    recognizer = sr.Recognizer()
    speech_lang = get_lang_config(language)["speech_rec"]

    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language=speech_lang)
    except sr.UnknownValueError:
        raise ValueError("Could not understand audio clearly. Please speak again.")
    except Exception as e:
        raise ValueError(f"Speech processing error: {str(e)}")


def text_to_speech(text: str, language: str) -> Optional[bytes]:
    """Generates MP3 audio bytes using gTTS."""
    if not text or not text.strip():
        return None

    gtts_lang = get_lang_config(language)["gtts"]
    try:
        tts = gTTS(text=text.strip(), lang=gtts_lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        raise RuntimeError(f"Text-to-Speech failed: {str(e)}")


@st.cache_resource(show_spinner="Loading OCR engine...")
def get_ocr_reader(lang_code: str) -> easyocr.Reader:
    """Instantiates and caches EasyOCR reader."""
    langs = list(set(["en", lang_code]))
    return easyocr.Reader(langs, gpu=False)


def extract_text_from_image(image_bytes: bytes, language: str) -> str:
    """Extracts text from uploaded image via EasyOCR."""
    if not image_bytes:
        raise ValueError("No image provided.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image.convert("RGB"))

        lang_code = get_lang_config(language)["easyocr"]
        reader = get_ocr_reader(lang_code)

        results = reader.readtext(image_np, detail=0)
        extracted = " ".join(results).strip()

        if not extracted:
            raise ValueError("No readable text found in image.")
        return extracted
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"OCR processing error: {str(e)}")


# ==============================================================================
# 3. STREAMLIT USER INTERFACE
# ==============================================================================

st.set_page_config(
    page_title="Local Slang Translator",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary-color: #2563eb;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .custom-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .slang-badge {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .custom-card {
            background-color: #1e293b;
            border-color: #334155;
        }
        .slang-badge {
            background-color: #1e3a8a;
            color: #93c5fd;
        }
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Session state initialization
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "translation_history" not in st.session_state:
    st.session_state.translation_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None


def main():
    st.title("🌐 Local Slang Translator")
    st.caption("Bridge slang barriers across English, Hindi, Korean, and Spanish.")

    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "⚠️ `GROQ_API_KEY` is missing. The app will operate using offline dictionary fallback."
        )

    # Sidebar Options
    with st.sidebar:
        st.header("⚙️ Configuration")
        language = st.selectbox("Select Language", list(LANGUAGES.keys()), index=0)
        mode = st.radio(
            "Translation Direction",
            ["Slang → Professional", "Professional → Slang"],
            index=0,
        )
        tone = st.selectbox(
            "Output Tone", ["Professional", "Formal", "Friendly-Neutral"], index=0
        )

        st.divider()
        st.header("📜 Session History")
        if st.button("Clear History"):
            st.session_state.translation_history = []
            st.rerun()

        if st.session_state.translation_history:
            for item in reversed(st.session_state.translation_history):
                with st.expander(f"{item['mode']} ({item['lang']})"):
                    st.write(f"**In:** {item['input']}")
                    st.write(f"**Out:** {item['output']}")
        else:
            st.info("No translations in this session yet.")

    # Two-Column Desktop Layout
    col_input, col_output = st.columns(2, gap="large")

    # LEFT COLUMN: INPUT
    with col_input:
        st.subheader("📥 Input")
        tab_text, tab_voice, tab_ocr = st.tabs(
            ["📝 Text", "🎙️ Voice Input", "📷 Image OCR"]
        )

        with tab_text:
            text_val = st.text_area(
                "Enter text to convert:",
                value=st.session_state.input_text,
                height=180,
                key="text_input_area",
                placeholder="Type or paste slang/text here...",
            )
            st.session_state.input_text = text_val

        with tab_voice:
            st.write("Record spoken voice:")
            audio_data = st.audio_input("Record audio", key="voice_recorder")
            if audio_data is not None and st.button("Transcribe Voice"):
                try:
                    with st.spinner("Transcribing audio..."):
                        transcription = transcribe_audio(audio_data.read(), language)
                        st.session_state.input_text = transcription
                        st.success("Audio transcribed!")
                        st.rerun()
                except Exception as err:
                    st.error(f"Voice Error: {str(err)}")

        with tab_ocr:
            st.write("Upload image (signboards, screenshots, menus):")
            uploaded_file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None and st.button("Extract Text from Image"):
                try:
                    with st.spinner("Extracting text via OCR..."):
                        extracted = extract_text_from_image(uploaded_file.read(), language)
                        st.session_state.input_text = extracted
                        st.success("Text extracted!")
                        st.rerun()
                except Exception as err:
                    st.error(f"OCR Error: {str(err)}")

        if st.button("🚀 Translate Text", type="primary"):
            if not st.session_state.input_text.strip():
                st.error("Please enter or record text to translate.")
            else:
                with st.spinner("Translating & Analyzing Slang..."):
                    result = translate_text(
                        st.session_state.input_text, language, mode, tone
                    )
                    st.session_state.current_result = result

                    st.session_state.translation_history.append(
                        {
                            "lang": language,
                            "mode": mode,
                            "input": st.session_state.input_text,
                            "output": result.get("translation", ""),
                        }
                    )
                    if len(st.session_state.translation_history) > 10:
                        st.session_state.translation_history.pop(0)

    # RIGHT COLUMN: OUTPUT
    with col_output:
        st.subheader("📤 Result")
        res = st.session_state.current_result

        if res:
            if "warning" in res:
                st.warning(res["warning"])

            translated_text = res.get("translation", "")
            st.text_area(
                "Translation Output:",
                value=translated_text,
                height=150,
                disabled=False,
                key="output_display",
            )

            if st.button("🔊 Read Aloud"):
                try:
                    audio_bytes = text_to_speech(translated_text, language)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.error(f"Audio Error: {str(e)}")

            st.subheader("💡 Detected Slang & Meaning")
            slang_items = res.get("detected_slang", [])
            if slang_items:
                for item in slang_items:
                    term = item.get("term", "N/A")
                    meaning = item.get("meaning", "N/A")
                    st.markdown(
                        f"""
                        <div class="custom-card">
                            <span class="slang-badge">{term}</span>
                            <span>{meaning}</span>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No slang terms detected.")
        else:
            st.info("Translation results will appear here after submission.")


if __name__ == "__main__":
    main()