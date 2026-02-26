from openai import OpenAI
#import config
# 1 - Set client
client = OpenAI(api_key=config.OPENAI_API_KEY)
# 2 - Create a file
batch_input_file = client.files.create(
        file=open("part_ae", "rb"),
  purpose="batch"
)
print(batch_input_file)
batch_input_file_id = batch_input_file.id
# 3 - Send a batch
batch = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)
# Save the batch id in a file
with open("output/batch_id.txt", "w") as f:
    f.write(batch.id)
# Save the batch object in a file
with open("output/batch.txt", "w") as f:
    f.write(str(batch))
