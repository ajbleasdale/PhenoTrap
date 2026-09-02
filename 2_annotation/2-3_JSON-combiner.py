# Example for combining 5 JSON files together
import json

json1 = r"path-to-JSON-1.json"
json2 = r"path-to-JSON-2.json"
json3 = r"path-to-JSON-3.json"
json4 = r"path-to-JSON-4.json"
json5 = r"path-to-JSON-5.json"
jsonX = r"path-to-JSON-5.json"


output = r"output-to-combined-json.json"

with open(json1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(json2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

with open(json3, "r", encoding="utf-8") as f:
    data3 = json.load(f)

with open(json4, "r", encoding="utf-8") as f:
    data4 = json.load(f)

with open(json5, "r", encoding="utf-8") as f:
    data5 = json.load(f)

with open(jsonx, "r", encoding="utf-8") as f:
    datax = json.load(f)

    # Both must be lists of tasks
    if not isinstance(data1, list) or not isinstance(data2, list):
     raise ValueError("Both JSON files must contain lists of tasks.")

    combined = data1 + data2 + data3 + data4 + data5 + datax
    with open(output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("Done: wrote combined JSON to", output)
