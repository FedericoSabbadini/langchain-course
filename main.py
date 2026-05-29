from typing import Any, Dict, List
import streamlit as st # to run this file, run `streamlit run main.py` in the terminal
from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
# st.set_page_config is used to set the title and layout of the app. It should be called at the very beginning of the script, before any other Streamlit commands.
st.title("LangChain Documentation Helper")
# st.title is used to display the title of the app on the main page. It should be called after set_page_config.
with st.sidebar: # st.sidebar is used to create a sidebar in the app. Any Streamlit commands inside this block will be displayed in the sidebar.
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        # st.button is used to create a button in the sidebar. When the button is clicked, it will clear the chat history by removing the "messages" key from the session state and rerunning the app.
        st.session_state.pop("messages", None)
        st.rerun()
        # st.session_state is a dictionary-like object that allows you to store and retrieve values across different runs of the app. In this case, we are using it to store the chat history in the "messages" key. When the "Clear chat" button is clicked, we remove this key from the session state, effectively clearing the chat history. Then we call st.rerun() to refresh the app and reflect the changes.
        # st.rerun() is used to rerun the app from the beginning. This is necessary after clearing the chat history to update the UI and show that the chat has been cleared.

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": ["(Example source).com"],
        }
    ]

for msg in st.session_state.messages:
    # we create a stored list of messages in the session state to keep track of the chat history. Each message is a dictionary with a "role" (either "user" or "assistant"), "content" (the text of the message), and optionally "sources" (a list of sources associated with the message). We loop through this list of messages and display each one in the app using st.chat_message.
    # st.chat_message is used to display a chat message in the app. The "role" key in the message dictionary determines the styling of the message (e.g., user messages are aligned to the right, assistant messages are aligned to the left). The "content" key contains the text of the message, which is displayed using st.markdown. If there are any sources associated with the message, they are displayed in an expander below the message.
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")
# st.chat_input is used to create a chat input box at the bottom of the app. The user can type their question here and submit it. The value of the input will be stored in the variable "prompt". When the user submits a question, we will process it and generate a response using the run_llm function.
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt) # we add the user's message to the session state messages list and display it in the chat using st.chat_message. The "role" is set to "user", the "content" is the prompt entered by the user, and "sources" is an empty list since user messages do not have associated sources.

    with st.chat_message("assistant"):
        # when the user submits a question, we create a new chat message with the role "assistant" to indicate that we are generating a response. We use st.spinner to show a loading spinner while we retrieve documents and generate the answer using the run_llm function. The result from run_llm is expected to be a dictionary containing an "answer" key with the generated answer and a "context" key with the relevant documents used as sources.
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)

