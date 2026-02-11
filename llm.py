from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

def load_llm(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    return HuggingFacePipeline(pipeline=pipe)
