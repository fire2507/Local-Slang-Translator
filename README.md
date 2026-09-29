#  Local Slang Translator

An AI-powered, multi-modal web application designed to bridge cultural and conversational language gaps. It translates informal slang, regional idioms, and local expressions into professional/standard language—and vice versa—across **English, Hindi, Korean, and Spanish**.

##  Problem Statement

Slang and regional dialects create significant communication barriers:
- **In Professional Settings:** International teams, remote workers, and cross-border communicators often misunderstand casual workplace idioms or struggle to rephrase colloquial messages into professional tones.
- **For Travelers & Language Learners:** Everyday expressions, street signs, menus, and spoken slang are rarely covered in traditional language apps or textbooks.

---

##  Solution

**Local Slang Translator** leverages LLMs (Groq / LLAMA 3.3) to provide instant, context-aware translations between informal slang and professional language. 

Beyond simple text-to-text translation, the application provides **structured linguistic breakdowns** explaining the exact slang terms used, and offers **multi-modal input/output** (Voice Recording, Image OCR, and Text-to-Speech) to handle real-world scenarios seamlessly.

##  Unique Features

1. **Bi-Directional Translation Engine:**
   - **Slang → Professional:** Converts casual expressions, idioms, and street talk into clean, professional prose.
   - **Professional → Slang:** Rephrases formal corporate language into authentic local slang.

2. **Structured Slang Detection & Explanation:**
   - Returns a dedicated **Explanation Panel** listing detected slang terms and their contextual meanings alongside the translation.

3. **Multi-Modal Accessibility:**
   - **Voice-to-Text Input:** Record speech directly via browser microphone for hands-free translation.
   - **Image OCR Reader:** Upload photos of street signs, printed menus, or text screenshots to extract and translate text.
   - **Text-to-Speech (TTS):** Listen to the translated output read aloud in native accents.

4. **Dynamic LLM Model Discovery & Fallback:**
   - Automatically queries active LLM endpoints at runtime, preventing API failure if models are decommissioned.
   - Built-in **Offline Dictionary Fallback** for core slang terms when no internet or API key is present.

5. **Session History Tracking:**
   - Retains a rolling log of your last 10 translations in a sidebar history drawer.


##  Tech Stack

- **UI Framework:** Streamlit
- **LLM Provider:** Groq API (LLAMA 3.3 / LLAMA 3.1)
- **Speech Recognition:** `SpeechRecognition` + `st.audio_input`
- **Optical Character Recognition (OCR):** `EasyOCR`
- **Text-to-Speech (TTS):** `gTTS` (Google Text-to-Speech)
- **Language & Runtime:** Python 3.10+
