from config import GEMINI_API_KEY
from google import genai


client = genai.Client(api_key=GEMINI_API_KEY)


class GenAIModelAdapter:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_content(self, prompt: str):
        return client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )


def configure_genai():
    pass


def make_model(model_name: str):
    return GenAIModelAdapter(model_name)


def generate_with_model(model, prompt: str):
    response = model.generate_content(prompt)

    if hasattr(response, "text"):
        return response.text

    return response