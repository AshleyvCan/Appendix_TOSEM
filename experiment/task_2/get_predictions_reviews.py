import pandas as pd
import json
import time
import os
import sys

from utils.generate_response import get_response
from utils.gpu_energy_recorder import GPURecorder

def run_models(data_reviews, models,  system_message, user_prompt, desc, app_name, export_settings):
    
    models_metrics = {}
    measurement_interval = 0.1  
    
    rounds = 1
    for b in [1, 4, 16, 64, 128]:
        for i in range(rounds):
            for model in models:
                model_name = model
                delimiter = model_name.find("_")
                energy_filename = f"{export_settings['energy']}/reviews{model_name[:delimiter]}_{app_name}"
                gpu_file_name = f"{energy_filename}_gpu.csv"
                gpu_recorder = GPURecorder(gpu_file_name,b,  measurement_interval)

                gpu_recorder.start()
                result, model_metr = get_response(model_name, data_reviews, system_message, user_prompt, desc, batch_size = b)
                gpu_recorder.stop()

                model_metr["model_name"] = model_name
                model_metr["round"] = i
                model_metr["batch_size"] = b
                model_metr["project"] = app_name

                models_metrics[f'{model_name}_r{i}_b{b}_{app_name}'] = model_metr


                eval_filename = f"{export_settings['response']}/reviews_{model_name[:delimiter]}_{app_name}.jsonl"
                
                with open(eval_filename, "ab") as outfile:
                    for data in result:
                        outfile.write((json.dumps(data) + "\n").encode('utf-8'))
                time.sleep(30)

    with open(f"{export_settings['energy']}/models_stats_reviews.jsonl", mode='a', newline='') as file:
        for key, value in models_metrics.items():
            json_line = json.dumps({key: value})
            file.write(json_line + '\n')

def get_project_info():
    with open("prompts/app_names.json", "r") as f:
        data = json.load(f)
    return data 


def main():
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

    models = ["llama3.1:8b-instruct-fp16", "gemma3:12b-it-fp16", "phi4:14b-fp16", "qwen3:8b-fp16"]
    
    app_names = [f for f in os.listdir("surminer_data") if f.endswith(".xlsx")]
    
    desc = get_project_info()

    with open(r"prompts/prompt_system_reviews.txt", "r") as file:
        system_prompt = file.read()

    with open(r"prompts/prompt_user_reviews.txt", "r") as file:
        user_prompt = file.read()


    for app_name in app_names:
        data = pd.read_excel("surminer_data/"+app_name)
        quotient = len(data) // 128   
        run_models(data.to_dict(orient="records")[:(quotient*128)], 
                   models,  
                   system_prompt, 
                   user_prompt,
                   desc[app_name.replace('.xlsx', '')],
                   app_name.replace('.xlsx', ''), 
                   export_settings)


if __name__ == "__main__":
    main()