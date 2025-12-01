import json
import random
place_meta = json.load(open("place_metadata.json"))
all_available_dining_places = set()
for place in place_meta:
    if place_meta[place]["coarse_type"] == "food":
        all_available_dining_places.add(place)
characters = json.load(open("gpt_cache/g4c15/characters.json"))
for agent_name in characters:
    for dining_place in characters[agent_name]["dining_places"]:
        if dining_place not in all_available_dining_places:
            characters[agent_name]["dining_places"].remove(dining_place)
            characters[agent_name]["dining_places"].append(
                random.choice(list(all_available_dining_places))
            )
with open("gpt_cache/g4c15/characters.json", "w") as f:
    json.dump(characters, f, indent=4)
print("Updated dining places for all characters.")