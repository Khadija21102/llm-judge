from openai import OpenAI
client = OpenAI(api_key=config.OPENAI_API_KEY)

# 2 - Read from batch_id.txt file
with open("output/batch_id.txt", "r") as file:
    batch_id = file.read()
# 3 - Check the batch status
batch_status = client.batches.retrieve(batch_id)
print(batch_status)
print("Batch output file: ", batch_status.output_file_id)

# 4 - Get the batch results
batch_output_id = batch_status.output_file_id
batch_output_file = client.files.content(batch_output_id) #print(batch_output_file.text)
# 5 - Save the output to a file
with open("output/api_response_pref_v2.jsonl", "a") as file:
    file.write(batch_output_file.text)
# 6 - Read the jsonl file
import json
with open("output/api_response_pref_v2.jsonl", "r") as file:
    api_response = file.readlines()
    api_response = [json.loads(line) for line in api_response]

# Print the first response as a json with indentation
print(json.dumps(api_response[0], indent=2))
# Cost estimation
# 9 - Calculate the number of tokens
def calculate_tokens(api_response):
    input_tokens = 0
    output_tokens = 0
    for i in range(len(api_response)):
        tokens = api_response[i]['response']['body']['usage']['prompt_tokens']
        input_tokens += tokens
        tokens = api_response[i]['response']['body']['usage']['completion_tokens']
        output_tokens += tokens
    return input_tokens, output_tokens
input_tokens, output_tokens = calculate_tokens(api_response)
print(f'Input tokens: {input_tokens} and {output_tokens} for {len(api_response)} messages')
from utils import load_data
#df_cases, df_labels, df_paths = load_data()
price_1M_tokens = 1.25
price_1M_tokens_output = 5

print(f'The cost for {input_tokens} tokens is: {input_tokens/1e6*price_1M_tokens}')
print(f'The cost for {output_tokens} tokens is: {output_tokens/1e6*price_1M_tokens_output}')
print(f'The total cost is: {input_tokens/1e6*price_1M_tokens + output_tokens/1e6*price_1M_tokens_output} for {len(api_response)} cases')
print(f'The cost per case is: {(input_tokens/1e6*price_1M_tokens + output_tokens/1e6*price_1M_tokens_output)/len(api_response)}')
#print(f'The cost for the whole dataset, {len(df_cases)} cases, is: {(input_tokens/1e6*price_1M_tokens + output_tokens/1e6*price_1M_tokens_output)/len(api_response)*len(df_cases)}')
