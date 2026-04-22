from app.core.config import settings
from app.llm.prompts import supported_llms

def load_llm(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5-20250929",
    temperature: int=0
):
    ''' Load and return the specified LLM '''
    
    if provider not in supported_llms:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Supported providers are: {list(supported_llms.keys())}")

    llm_class = supported_llms[provider]
    
    return llm_class(model=model, temperature=temperature, api_key=settings.anthropic_api_key)
