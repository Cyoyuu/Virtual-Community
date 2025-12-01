import argparse
import json

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--result", "-r", type=str, required=True)
    args=parser.parse_args()
    data=json.load(open(args.result, "r"))
    output=""
    agent_type_match = {
        "center_no_avoidance": "Oracle Centered (No Danger Zone)",
        "center": "Oracle Centered",
        "roco": "RoCo",
        "coela": "Coela",
        "sentinel": "SENTINEL",
        "sentinel_no_refine": "SENTINEL w/o refine"
    }
    for agent_type in agent_type_match:
        if agent_type in data:
            # num=len(data[agent_type].keys())-1
            # if agent_type=="sentinel":num=16
            num=16
            desc=f"{agent_type_match[agent_type]} & {(data[agent_type]['average']['success_rate']*100*num/16):.2f} & {(data[agent_type]['average']['caught_rate']*100):.2f} & {(data[agent_type]['average']['detection_rate']*100):.2f} & {data[agent_type]['average']['time_spent_meeting_mean']:.2f} & {data[agent_type]['average']['walk_spent_meeting_mean']:.2f} \\\\ \n"
        else:
            desc=f"{agent_type_match[agent_type]} & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\\\ \n"
        output=output+desc
    print(output)