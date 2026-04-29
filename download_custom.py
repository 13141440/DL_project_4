"""Download extra choose_img samples from lmms-lab/ICON-QA to build custom.arrow."""

import pyarrow as pa
from datasets import load_dataset, Dataset, Features, Value, Image

# Load the val split from lmms-lab/ICON-QA (21488 samples, same schema as our training data)
print("Downloading lmms-lab/ICON-QA val split...")
ds = load_dataset("lmms-lab/ICON-QA", split="val")

# Filter for choose_img type only (matching our training data format)
print(f"Total samples: {len(ds)}")
ds_filtered = ds.filter(
    lambda x: x["ques_type"] == "choose_img"
    and x["choice_image_0"] is not None
    and x["choice_image_1"] is not None
)
print(f"choose_img samples: {len(ds_filtered)}")

# Get existing training question_ids to avoid duplicates
train_ds = Dataset.from_file("data/icon-qa-train.arrow")
train_ids = set(train_ds["question_id"])
val_ds = Dataset.from_file("data/icon-qa-val.arrow")
val_ids = set(val_ds["question_id"])
exclude_ids = train_ids | val_ids

ds_filtered = ds_filtered.filter(lambda x: x["question_id"] not in exclude_ids)
print(f"After removing duplicates with train/val: {len(ds_filtered)}")

# Take at most 1000 samples (the maximum allowed)
if len(ds_filtered) > 1000:
    ds_filtered = ds_filtered.select(range(1000))
print(f"Final custom dataset size: {len(ds_filtered)}")

# Keep only the columns needed by our convert_custom_train_to_conversation
keep_cols = ["question", "choices", "answer", "query_image", "choice_image_0", "choice_image_1"]
ds_out = ds_filtered.select_columns(keep_cols)

# Flatten indices to materialize the filtered/selected subset into a contiguous table
ds_out = ds_out.flatten_indices()

# Save as IPC stream Arrow format (compatible with datasets.Dataset.from_file)
table = ds_out.data
sink = pa.OSFile("custom.arrow", "wb")
writer = pa.ipc.new_stream(sink, table.schema)
for batch in table.to_batches():
    writer.write_batch(batch)
writer.close()
sink.close()

# Verify
loaded = Dataset.from_file("custom.arrow")
print(f"\ncustom.arrow saved successfully: {len(loaded)} samples")
print(f"Columns: {loaded.column_names}")
print(f"Sample question: {loaded[0]['question']}")
print(f"Sample answer: {loaded[0]['answer']}")
