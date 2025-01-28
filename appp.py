import validators
import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFaceInference
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain.docstore.document import Document

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Get the URL to be summarized
generic_url = st.text_input("URL", label_visibility="collapsed")

# Initialize the T5 model
hugging_face_api_key = "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"  # Get a free API key from Hugging Face
llm = HuggingFaceInference(
    repo_id="t5-base",
    max_tokens=2048,
    api_key=hugging_face_api_key,
)

# Prompt template for summarization
prompt_template = """
Provide a detailed summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    # Validate all the inputs
    if not hugging_face_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                docs = None
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    # Extract the video ID from the generic URL
                    video_id = (
                        generic_url.split("v=")[-1].split("&")[0]
                        if "v=" in generic_url
                        else generic_url.split("/")[-1]
                    )
                    try:
                        # Attempt to fetch the English transcript
                        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                    except Exception:
                        st.warning("Manual English transcript not found. Attempting auto-generated English transcript...")
                        try:
                            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["a.en"])
                        except Exception:
                            st.warning("English transcript unavailable. Checking for available subtitles in other languages...")
                            try:
                                # Check if there are other available transcripts
                                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                                # Get the first available transcript
                                transcript = transcript_list.find_transcript(
                                    [lang.language_code for lang in transcript_list]
                                ).fetch()
                            except Exception as e:
                                if "Subtitles are disabled for this video" in str(e):
                                    st.error("Subtitles are disabled for this video. No transcript is available.")
                                else:
                                    st.error(f"Error fetching transcript: {e}")
                                transcript = None

                    if transcript:
                        # Format and process the transcript
                        formatter = TextFormatter()
                        transcript_text = formatter.format_transcript(transcript)
                        docs = [Document(page_content=transcript_text)]
                    else:
                        docs = None
                else:
                    # For non-YouTube URLs
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )
                    docs = loader.load()

                # Chain for Summarization
                if docs:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    st.success(output_summary)
                else:
                    st.error("No content could be extracted from the provided URL.")
        except Exception as e:
            st.exception(f"Exception: {e}")
