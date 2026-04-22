from langchain_anthropic import ChatAnthropic
from langchain_classic import hub 

supported_prompts = {
    "rag" : "rlm/rag-prompt"
}

supported_llms = {
    "anthropic" : ChatAnthropic
}

def pull_rag_prompt(prompt_name: str = "rlm/rag-prompt"):
    ''' Pull and return the specified prompt '''

    if prompt_name not in supported_prompts:
        raise ValueError(f"Unsupported prompt name '{prompt_name}'. Supported prompts are: {list(supported_prompts.keys())}")
    
    return hub.pull(supported_prompts[prompt_name])

