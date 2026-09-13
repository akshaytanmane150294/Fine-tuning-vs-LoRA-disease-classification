from datasets import load_dataset, concatenate_datasets

raw = load_dataset('gretelai/symptom_to_diagnosis')
print(raw)

# merge all splits into one, so the count is accurate regardless of
# whether classes are split across train/test differently
all_df = concatenate_datasets([raw[s] for s in raw.keys()]).to_pandas()

print("\nColumns:", all_df.columns.tolist())

# try the most likely label column name first
label_col = "output_text" if "output_text" in all_df.columns else None
if label_col is None:
    print("\n[!] 'output_text' column not found - check the printed columns above "
          "and set label_col manually.")
else:
    unique_labels = sorted(all_df[label_col].unique())
    print(f"\nnum_classes = {len(unique_labels)}")
    print("\nAll disease labels:")
    for label in unique_labels:
        print(" -", label)
