import streamlit as st
import re
import json
import io

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
        "gemini-2.5-flash"
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
7. The interviewer may use very short transitions.
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
# TEXT TO SPEECH - MP3
# ---------------------------------------------------

def synthesize_mp3(
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
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content


# ---------------------------------------------------
# COMBINE MP3 CHUNKS
# ---------------------------------------------------

def combine_mp3_files(audio_chunks):
    """
    Google TTS returns complete MP3 chunks.
    Concatenating MP3 streams works for playback/download
    when all chunks use the same encoding settings.
    """
    return b"".join(audio_chunks)


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

A: I enjoy technology and helping people.
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

                # Interviewer
                interviewer_audio = synthesize_mp3(
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

                # Candidate
                candidate_audio = synthesize_mp3(
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

        # Combine all MP3 pieces
        final_audio = combine_mp3_files(
            audio_chunks
        )

        progress.progress(1.0)

        st.success(
            "Interview generated successfully!"
        )

        # ---------------------------------------------------
        # AUDIO PLAYER
        # ---------------------------------------------------

        st.markdown("## 🎧 Your Interview")

        st.audio(
            final_audio,
            format="audio/mp3"
        )

        # ---------------------------------------------------
        # DOWNLOAD MP3
        # ---------------------------------------------------

        st.download_button(
            "⬇️ Download Interview MP3",
            data=final_audio,
            file_name="interview_practice.mp3",
            mime="audio/mpeg",
            use_container_width=True
        )

        # ---------------------------------------------------
        # FINAL TEXT
        # ---------------------------------------------------

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
