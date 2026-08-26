import streamlit as st
import re
import json
import io
import wave

from google.cloud import texttospeech
from google.oauth2 import service_account

# Gemini is optional
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


st.set_page_config(
    page_title="Interview Conversation Generator",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Interview Conversation Generator")
st.write(
    "Upload your interview questions and answers and turn them into "
    "a two-person audio conversation."
)


# ---------------------------------------------------
# GOOGLE CLOUD AUTHENTICATION
# ---------------------------------------------------

def get_tts_client():
    try:
        credentials_info = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"],
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url":
                st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url":
                st.secrets["gcp_service_account"]["client_x509_cert_url"],
        }

        credentials = service_account.Credentials.from_service_account_info(
            credentials_info
        )

        return texttospeech.TextToSpeechClient(credentials=credentials)

    except Exception as e:
        st.error("Google Cloud credentials are missing or incorrect.")
        st.code(str(e))
        return None


# ---------------------------------------------------
# Q&A PARSER
# ---------------------------------------------------

def parse_qa(text):
    """
    Expected format:

    Q: Tell me about yourself.
    A: My name is...

    Q: Why do you want this job?
    A: Because...
    """

    pattern = r"""
        (?:^|\n)\s*
        (?:Q|Question)\s*[:\-]\s*
        (.*?)
        \n\s*
        (?:A|Answer)\s*[:\-]\s*
        (.*?)
        (?=
            \n\s*(?:Q|Question)\s*[:\-]
            |\Z
        )
    """

    matches = re.findall(
        pattern,
        text,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    pairs = []

    for question, answer in matches:
        pairs.append({
            "question": question.strip(),
            "answer": answer.strip()
        })

    return pairs


# ---------------------------------------------------
# GEMINI NATURALIZATION
# ---------------------------------------------------

def make_natural_with_gemini(pairs):
    if "GEMINI_API_KEY" not in st.secrets:
        raise Exception(
            "GEMINI_API_KEY is not configured in Streamlit Secrets."
        )

    if not GEMINI_AVAILABLE:
        raise Exception("google-genai package is not installed.")

    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    model_name = st.secrets.get(
        "GEMINI_MODEL",
        "gemini-3.7-flash"
    )

    prompt = f"""
You are preparing audio for a job interview practice session.

Turn the following interview questions and answers into a natural
two-person interview conversation.

IMPORTANT RULES:

1. Keep every original question.
2. Preserve the meaning and factual content of every answer.
3. Do NOT invent work experience, skills, companies, achievements,
   education, or personal information.
4. Do NOT remove important information from the answers.
5. You may only make small conversational improvements.
6. Keep the candidate's language simple and natural.
7. The interviewer may use very short transitions such as:
   "Great, thank you."
   "Can you tell me about..."
   "Okay. What about..."
8. Do not turn it into a podcast.
9. This is a realistic job interview practice session.

Return ONLY valid JSON in this exact structure:

[
  {{
    "question": "...",
    "answer": "..."
  }}
]

Interview content:

{json.dumps(pairs, ensure_ascii=False, indent=2)}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    result = json.loads(response.text)

    if not isinstance(result, list):
        raise Exception("Gemini returned an unexpected response.")

    return result


# ---------------------------------------------------
# TEXT TO SPEECH
# ---------------------------------------------------

def synthesize_wav(
    client,
    text,
    voice_name,
    language_code="en-US"
):
    synthesis_input = texttospeech.SynthesisInput(
        text=text
    )

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content


# ---------------------------------------------------
# WAV COMBINATION
# ---------------------------------------------------

def combine_wav_files(audio_chunks, pause_ms=500):
    output = io.BytesIO()

    first_audio = io.BytesIO(audio_chunks[0])

    with wave.open(first_audio, "rb") as first_wave:
        channels = first_wave.getnchannels()
        sample_width = first_wave.getsampwidth()
        frame_rate = first_wave.getframerate()

    with wave.open(output, "wb") as output_wave:
        output_wave.setnchannels(channels)
        output_wave.setsampwidth(sample_width)
        output_wave.setframerate(frame_rate)

        silence_frames = int(
            frame_rate * pause_ms / 1000
        )

        silence = (
            b"\x00"
            * silence_frames
            * channels
            * sample_width
        )

        for chunk in audio_chunks:
            audio_io = io.BytesIO(chunk)

            with wave.open(audio_io, "rb") as wav_file:
                frames = wav_file.readframes(
                    wav_file.getnframes()
                )

                output_wave.writeframes(frames)
                output_wave.writeframes(silence)

    output.seek(0)

    return output.getvalue()


# ---------------------------------------------------
# FORMAT CONVERSATION
# ---------------------------------------------------

def conversation_to_text(pairs):
    text = ""

    for i, pair in enumerate(pairs, start=1):
        text += f"""
Question {i}

Interviewer:
{pair["question"]}

Candidate:
{pair["answer"]}

{"-" * 50}

"""

    return text


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Conversation style",
    [
        "Exact Q&A",
        "Natural Interview"
    ]
)

interviewer_voice = st.sidebar.selectbox(
    "Interviewer voice",
    [
        "en-US-Chirp3-HD-Charon",
        "en-US-Chirp3-HD-Orus",
        "en-US-Chirp3-HD-Puck",
        "en-US-Chirp3-HD-Alnilam"
    ]
)

candidate_voice = st.sidebar.selectbox(
    "Candidate voice",
    [
        "en-US-Chirp3-HD-Kore",
        "en-US-Chirp3-HD-Aoede",
        "en-US-Chirp3-HD-Zephyr",
        "en-US-Chirp3-HD-Despina"
    ]
)

pause_between = st.sidebar.slider(
    "Pause between speakers",
    min_value=200,
    max_value=1500,
    value=500,
    step=100
)


# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your interview TXT file",
    type=["txt"]
)


st.markdown("### Expected file format")

st.code(
"""Q: Tell me about yourself.

A: My name is Anas. I have a background in technology and customer service.

Q: Why do you want to work here?

A: I enjoy technology and helping people. I like helping people become more comfortable with technology.
""",
    language="text"
)


# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------

if uploaded_file is not None:

    text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    pairs = parse_qa(text)

    if not pairs:
        st.error(
            "I couldn't detect any questions and answers. "
            "Use Q: and A: before each question and answer."
        )
        st.stop()

    st.success(
        f"Found {len(pairs)} interview questions."
    )

    st.markdown("## 📝 Interview Preview")

    for i, pair in enumerate(pairs, start=1):
        with st.expander(
            f"Question {i}: {pair['question'][:70]}"
        ):
            st.markdown("**Interviewer**")
            st.write(pair["question"])

            st.markdown("**Candidate**")
            st.write(pair["answer"])

    if st.button(
        "🎙️ Generate Interview",
        type="primary",
        use_container_width=True
    ):

        final_pairs = pairs

        # Natural mode
        if mode == "Natural Interview":

            with st.spinner(
                "Making the interview sound more natural..."
            ):

                try:
                    final_pairs = make_natural_with_gemini(
                        pairs
                    )

                except Exception as e:
                    st.warning(
                        "Gemini could not process the interview. "
                        "Using the original Q&A instead."
                    )
                    st.code(str(e))
                    final_pairs = pairs

        client = get_tts_client()

        if client is None:
            st.stop()

        audio_chunks = []

        progress = st.progress(0)

        total_parts = len(final_pairs) * 2
        current_part = 0

        try:

            for pair in final_pairs:

                interviewer_audio = synthesize_wav(
                    client,
                    pair["question"],
                    interviewer_voice
                )

                audio_chunks.append(
                    interviewer_audio
                )

                current_part += 1
                progress.progress(
                    current_part / total_parts
                )

                candidate_audio = synthesize_wav(
                    client,
                    pair["answer"],
                    candidate_voice
                )

                audio_chunks.append(
                    candidate_audio
                )

                current_part += 1
                progress.progress(
                    current_part / total_parts
                )

        except Exception as e:
            st.error(
                "Google Text-to-Speech returned an error."
            )
            st.code(str(e))
            st.stop()

        with st.spinner(
            "Combining interview audio..."
        ):
            final_audio = combine_wav_files(
                audio_chunks,
                pause_ms=pause_between
            )

        progress.progress(1.0)

        st.success(
            "Interview generated successfully!"
        )

        st.markdown("## 🎧 Your Interview")

        st.audio(
            final_audio,
            format="audio/wav"
        )

        st.download_button(
            "⬇️ Download Interview",
            data=final_audio,
            file_name="interview_practice.wav",
            mime="audio/wav",
            use_container_width=True
        )

        st.markdown("## 💬 Final Conversation")

        final_text = conversation_to_text(
            final_pairs
        )

        st.text_area(
            "Conversation",
            final_text,
            height=500
        )

        st.download_button(
            "⬇️ Download Conversation Text",
            data=final_text,
            file_name="interview_conversation.txt",
            mime="text/plain",
            use_container_width=True
        )
