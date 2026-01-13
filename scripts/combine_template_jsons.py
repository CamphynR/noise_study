import json
import os







if __name__ == "__main__":
    templates_older_path = "sim/library/deep_impulse_responses.json"
    with open(templates_older_path, "r") as templates_older_file:
        templates_older_content = json.load(templates_older_file)

    print(templates_older_content.keys())
    templates_newer_path = "sim/library/v2_v3_deep_impulse_responses_for_comparison.json"

    with open(templates_newer_path, "r") as templates_newer_file:
        templates_newer_content = json.load(templates_newer_file)
    print(templates_newer_content.keys())

    assert templates_older_content["time"] == templates_newer_content["time"]

    templates_older_content.pop("time", None)
    for key, content in templates_older_content.items():
        templates_newer_content[key] = content
    print(templates_newer_content.keys())


    jsons_dir = "sim/library" 
    new_json_name = "deep_templates_combined.json"
    save_path = os.path.join(jsons_dir, new_json_name)

    
    with open(save_path, "w") as file:
        json.dump(templates_newer_content, file, indent=4)
