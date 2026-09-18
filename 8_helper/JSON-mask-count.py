import json
import os
from collections import Counter

# ======================
# Path to JSON
# ======================
json_file = r"path-to-JSON-file.json"

# ======================
# Load JSON
# ======================
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# ======================
# Count individual polygon labels
# ======================
label_counter = Counter()
total_labels = 0

for task in data:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            value = result.get("value", {})

            if "polygonlabels" in value:
                for label in value["polygonlabels"]:
                    label_counter[label] += 1
                    total_labels += 1

# ======================
# Print results
# ======================
print(f"Total individual labels: {total_labels}\n")

print("Labels per class:")
for label, count in sorted(label_counter.items()):
    print(f"{label}: {count}")
