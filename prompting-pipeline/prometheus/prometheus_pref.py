# Relative Grading: Outputs A or B
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model_path= "prometheus-eval"

model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,
    load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

RELATIVE_PROMPT = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_A}

###Response B:
{response_B}


###Feedback: """

with open("data_with_ref_scores_preference_v2.jsonl", "r") as f, open("outputs_prometheus_pref.jsonl", "a") as f_out:
    for line in f:
        i = json.loads(line)
        user_content = ABS_SYSTEM_PROMPT + "\n\n" + RELATIVE_PROMPT.format(instruction=i["instruction"],
                response_A=i["response_A"],
                response_B=i["response_B"],
                rubric=i["rubric"],
                gold_response= i["gold_response"]
        ) # Fill the prompt with your data

        messages = [
            {"role": "user", "content": user_content},
        ]
        device = next(model.parameters()).device
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

        model_inputs = encodeds

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True, top_p=0.9, temperature=0.1, repetition_penalty=1.03)
        decoded = tokenizer.batch_decode(generated_ids)
        output_entry = {
            "instruction": i["instruction"],
            "response_A": i["response_A"],
            "response_B": i["response_B"],
            "gold_response": i["gold_response"],
            "generated_output": decoded[0]
        }
        f_out.write(json.dumps(output_entry) + "\n")



# Output
# Feedback: Both Response A and Response B correctly identify economic troubles and overreliance on slave labor as significant contributing factors to the fall of the Roman Empire. However, Response B is more effective in presenting the historian's argument due to its inclusion of scholarly sources to back up its claims. Specifically, it references works by Harper, Scheidel, and Temin, which adds credibility to the historian's argument and aligns well with the score rubric's emphasis on evidence and citations. While Response A provides a similar argument, it lacks any form of citations or attributions, which lessens the strength of the evidence presented. Therefore, based on the provided rubric, Response B is the superior response due to its use of scholarly evidence to support the historian's claims.
# Score:

