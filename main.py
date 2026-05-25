from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
import os

key = os.getenv("ANTHROPIC_API_KEY")


def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
