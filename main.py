from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


if __name__ == "__main__":
    load_dotenv()
    chat_model = init_chat_model()
    response = chat_model("What is the capital of France?")
    print(response)