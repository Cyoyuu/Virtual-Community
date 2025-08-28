import os
import json
import argparse

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str)
    args=parser.parse_args()
    output={}
    with open(args.file, "r") as f:
        steps=json.load(f)
        for step in steps:
            for key in steps[step]["action"]:
                if key not in output:
                    output[key]={"None": 0}
                if steps[step]["action"][key] is None:
                    output[key]["None"]+=1
                    continue
                if steps[step]["action"][key]["type"] not in output[key]:
                    output[key][steps[step]["action"][key]["type"]]=0
                output[key][steps[step]["action"][key]["type"]]+=1
    json.dump(output, open("check_action.json", "w"), indent=2)