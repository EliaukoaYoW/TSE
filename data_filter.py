import time
import json
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


def generate_single_turn(id: int, tweet: str, target: str, stance: str):

    json_template = (
        '{{"id": {},"fluency": float (0-1),"clarity": float (0-1),"leakage_severity": int (0 or 1),"sanitized_inferability": float (0-1),"stance_support": float (0-1),"ambiguity": float (0-1)}}'
    ).format(id) 
    messages = [
        {
            "role": "system",
            "content": 
            (
                "You are an expert annotator for stance detection data quality. "
                "Your task is to evaluate whether a given tweet expresses a clear and reasonable stance toward the specified target.\n"
                "Carefully follow the instructions and output a strict JSON object with the required fields.\n"
                "1. Fluency: Is the text grammatically correct and fluent to read? (0–1, continuous; higher = more fluent, lower = less fluent)\n"
                "2. Clarity: Is the meaning of the text clear and understandable, even with sarcasm/irony? (0–1, continuous; higher = clearer, lower = harder to interpret)\n"
                "3. Leakage Severity: Does the text explicitly reveal the target (hashtags, @mentions, quoted names)? (0 or 1, binary; 1 = leakage present [bad], 0 = no leakage [good])\n"
                "4. Sanitized Inferability: After removing leakage, can the target still be reasonably inferred? (0–1, continuous; higher = easier to infer, lower = impossible to infer)\n"
                "5. Stance Support: Does the text provide enough evidence for its stance? (0–1, continuous; higher = strong stance evidence, lower = weak or no evidence)\n"
                "6. Ambiguity: Is the target/stance ambiguous or multi-target? (0–1, continuous; higher = more ambiguous [bad], lower = unambiguous [good])\n"
                "Please output your judgment in the following JSON format and do not add explanations or extra text:\n"
                f"{json_template}"
            )
        },
        {
            "role": "user",
            "content": prefill_prompt(tweet, target, stance)
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128,temperature=0.6)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

results = []
batch_size = 10
output_file = "annotations.jsonl"

file = "pstance-trump-all.csv"
df = pandas.read_csv(file,nrows=1000)

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

start_time = time.time()
for row in range(df.shape[0]):
    id = df.iloc[row]['ID']
    tweet = df.iloc[row]['Tweet']
    target = df.iloc[row]['Target']
    stance = df.iloc[row]['Stance']
    response = generate_single_turn(id,tweet,target,stance)
    print(response)

    try:
        parsed = json.loads(response)
        if "id" not in parsed:
            parsed["id"] = id
    except Exception:
        parsed = {"id": id, "raw_response": response}
    results.append(parsed)

    if id % batch_size == 0:
        print("Current ID: ",id)
        with open(output_file, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results = []  # 清空缓存

    # 收尾：如果最后不足100条，也要写一次

if results:
    with open(output_file, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved remaining {len(results)} samples to {output_file}")

end_time = time.time()
print(f"Total time consumed:{int((end_time-start_time)/60)}")



