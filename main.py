import json
from collections import defaultdict


def jsonl_reader(file_name: str)-> list[dict]:
    data =[]
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data



def observation_detail_format(data:str)->list[dict]:
    data_list = json.loads(data)
    for item in data_list:
        item['id'] = int(item['id'])
    return data_list



def id_transcript_observ_extractor(grip:str)->list[str]:
     # dict structure: id/transcript/observation
    id = int(grip["id"] ) # str
    observations = grip['observations']  # str
    transcript = grip['transcript']  # str
    return id,transcript,observations



def data_read_main(file:str) -> dict:
    raw_data = jsonl_reader(file)  # list of dicts : len 101
    data_dict = defaultdict(list)
    for items in raw_data:
        binary_transc_observ_list = []
        visit_id, trans, obs = id_transcript_observ_extractor(items)
        observations_pack = observation_detail_format(obs)
        binary_transc_observ_list.append(trans)
        binary_transc_observ_list.append(observations_pack)
        data_dict[visit_id] = binary_transc_observ_list
    return data_dict



def schema_read(file_path:str):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    return raw_data


if __name__ == '__main__':

    from pathlib import Path
    script_dir = Path(__file__).parent
    data_path = script_dir / "Data" / "dev.jsonl"
    schema_path = script_dir / "Data" / "synur_schema.json"
    
    dataset = data_read_main(data_path)
    print(dataset[152][0])
    print(dataset[152][1])
    for i in dataset[152][1]:
        print(i['name'])
    schema = schema_read(schema_path)






# dataset is a list of dicts; each dict has three keys: id, transcripts, observation
# each observation is a list of dicts such that each dict is made up of 4 keys.








