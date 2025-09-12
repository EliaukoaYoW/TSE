import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def prefill_prompt(tweet,target,stance):
    content = f"Please analyze the following input: \n" \
              f"Tweet:{tweet}\n" \
              f"Target: {target}\n" \
              f"Stance: {stance}\n"
    return content


def generate_single_turn(tweet: str,target: str, stance: str):
    messages = [
        {
            "role": "system",
            "content": "You are an expert annotator for stance detection data quality. "
                       "Your task is to evaluate whether a given tweet expresses a clear and reasonable stance toward the specified target.\n"
                       "Carefully follow the instructions and output a strict JSON object with the required fields.\n"
                       "1. Fluency: Is the text grammatically correct and fluent to read? (0–1, continuous; higher = more fluent, lower = less fluent)\n"
                       "2. Clarity: Is the meaning of the text clear and understandable, even with sarcasm/irony? (0–1, continuous; higher = clearer, lower = harder to interpret)\n"
                       "3. Leakage Severity: Does the text explicitly reveal the target (hashtags, @mentions, quoted names)? (0 or 1, binary; 1 = leakage present [bad], 0 = no leakage [good])\n"
                       "4. Sanitized Inferability: After removing leakage, can the target still be reasonably inferred? (0–1, continuous; higher = easier to infer, lower = impossible to infer)\n"
                       "5. Stance Support: Does the text provide enough evidence for its stance? (0–1, continuous; higher = strong stance evidence, lower = weak or no evidence)\n"
                       "6. Ambiguity: Is the target/stance ambiguous or multi-target? (0–1, continuous; higher = more ambiguous [bad], lower = unambiguous [good])\n"
                       "Please output your judgment in the following JSON format:{'fluency': float (0-1),'clarity': float (0-1),'leakage_severity': float (0-1),'sanitized_inferability': float (0-1),'ambiguity': float (0-1)} Do not add explanations or extra text."
        },
        {
            "role": "user",
            "content": prefill_prompt(tweet, target, stance)
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response


file = "pstance-trump-all.csv"
df = pandas.read_csv(file,nrows=2)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

for row in range(df.shape[0]):
    tweet = df.iloc[row]['Tweet']
    target = df.iloc[row]['Target']
    stance = df.iloc[row]['Stance']
    response = generate_single_turn(tweet,target,stance)
    print(response)



