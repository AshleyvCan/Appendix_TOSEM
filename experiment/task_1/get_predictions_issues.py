import pandas as pd
import json
import time
import os
import sys

from utils.generate_response import get_response
from utils.gpu_energy_recorder import GPURecorder

def run_models(data_issues, models, system_message, user_prompt, desc, project_name, export_settings):
 
    models_metrics = {}
    measurement_interval = 0.1  
    
    rounds = 1
    for b in [1, 4, 16, 64, 128]:
        for i in range(rounds):
            for model in models:
                model_name = model
                delimiter = model_name.find("_")
                energy_filename = f"{export_settings['energy']}/ITS_{model_name[:delimiter]}_{project_name}"
                gpu_file_name = f'{energy_filename}_gpu.csv'
                gpu_recorder = GPURecorder(gpu_file_name, b, measurement_interval)

                gpu_recorder.start()
                result, model_metr = get_response(model_name, data_issues, system_message, user_prompt, desc, batch_size = b)
                gpu_recorder.stop()

                model_metr["model_name"] = model_name
                model_metr["round"] = i
                model_metr["batch_size"] = b
                model_metr["project"] = project_name

                models_metrics[f'{model_name}_r{i}_b{b}_{project_name}'] = model_metr
                
                eval_filename = f"{export_settings['response']}/ITS_{model_name[:delimiter]}_{project_name}.jsonl"
                
                with open(eval_filename, "ab") as outfile:
                    for data in result:
                        outfile.write((json.dumps(data) + "\n").encode('utf-8'))
                time.sleep(30)

    with open(f"{export_settings['energy']}/models_stats_ITS.jsonl", mode='a', newline='') as file:
        for key, value in models_metrics.items():
            json_line = json.dumps({key: value})
            file.write(json_line + '\n')


def get_project_info():
    with open("prompts/project_names.json", "r") as f:
        data = json.load(f)
    return data 


def main():

    models = ["llama3.1:8b-instruct-fp16", "gemma3:12b-it-fp16", "phi4:14b-fp16", "qwen3:8b-fp16"]
    
    data = pd.read_excel('<dataset_name>')

    project_names = set(data["project"])

    desc = get_project_info()

    export_folder = ''
    export_settings = {}
    if len(sys.argv) > 1:
        if sys.argv[1]: 
            os.makedirs(sys.argv[1], exist_ok=True)
            export_folder = sys.argv[1]

    export_settings['energy'] = os.path.join(export_folder, "energy")
    os.makedirs(export_settings['energy'], exist_ok=True)

    export_settings['response'] = os.path.join(export_folder, "response")
    os.makedirs(export_settings['response'], exist_ok=True)         

    data =data[data['text'].notna() & (data['text'].str.strip() != "")]

    with open(r"prompts/prompt_system_issues.txt", "r") as file:
        system_prompt = file.read()
    
    with open(r"prompts/prompt_user_issues.txt", "r") as file:
        user_prompt = file.read()

    data= data.rename(columns={"text": "segment"})
    
    for p in project_names:
        data_project = data[data['project'] == p]
        
        quotient = len(data_project) // 128   

        run_models(data_project.to_dict(orient="records")[:(quotient*128)], 
                   models,  
                   system_prompt,
                   user_prompt, desc[p],
                   p,
                   export_settings)


if __name__ == "__main__":
    main()