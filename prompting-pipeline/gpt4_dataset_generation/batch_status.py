from openai import OpenAI
client = OpenAI(api_key=config.OPENAI_API_KEY)

batch = client.batches.retrieve("batch_68e7bca686648190874ec37f09a45e5e")
print(batch)
print("Batch status: ", batch.status)



# List all batches (most recent first)
batches = client.batches.list()

print(f"Found {len(batches.data)} batches\n")

for b in batches.data:
    print(f"🆔 {b.id}")
    print(f"   Status: {b.status}")
    print(f"   Model:  {b.endpoint}")
    print(f"   Created: {b.created_at}")
    print(f"   Input file:  {b.input_file_id}")
    print(f"   Output file: {b.output_file_id}")
    print("------")
