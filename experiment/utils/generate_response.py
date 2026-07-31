import ollama
import time
from datetime import datetime

def generate_response(model_name, prompt, system_message, settings):
    return ollama.generate(model=model_name, prompt=prompt, stream=False, options=settings, system=system_message)

def unload_model(model_name):
    return ollama.generate(model=model_name, keep_alive=0)

def preload_model(model_name):
    return ollama.generate(model=model_name, keep_alive=-1)

def warm_up(model_name, data, system_prompt, user_prompt, desc, options):
    user_prompt, system_prompt = adjust_prompt_b1(user_prompt, system_prompt)
    for start_i in range(0, 10):
        batch_segments = str(start_i) +". "+ data[start_i]["segment"]
        batch_prompt = user_prompt+'\n'+ batch_segments
        response = generate_response(model_name, batch_prompt, system_prompt.format(project = desc, m=start_i), options)

def adjust_prompt_b1(user_prompt, system_prompt):
    system_prompt = system_prompt.replace("{{ {m}: <label>,   ... {n}: <label> }}", "{{{m}: <label>}}") 
    if "review" in user_prompt:
        user_prompt = "Here is the app review sentence:"
    else:
        user_prompt = "Here is the issue segment:"
    
    return user_prompt, system_prompt

def get_response(model_name, data, system_prompt,user_prompt, desc, batch_size = 1):
    settings = {
    "temperature": 0.0001,
    "seed": 42,
     "num_predict": 10000,
    "num_ctx": 10000,
    }

    output_responses = []

    model_metrics = {
        "done": 0,
        "unfinished_requests": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "prompt_eval_duration": 0,
        "eval_duration": 0,
        "total_duration": 0,
        "tokens_per_sec": 0,
        "total_elapsed_time": 0,
        "model_load_time": 0,
        "model_unload_time": 0,
        "time_stamp": str(datetime.now())
    }
    
    start_load_time = time.time()
    start_load_timestamp = datetime.now().isoformat(timespec='milliseconds')
    preload_model(model_name)
    end_load_timestamp = datetime.now().isoformat(timespec='milliseconds')
    end_load_time = time.time()
    

    start_warmup_time = time.time()
    start_warmup_timestamp = datetime.now().isoformat(timespec='milliseconds')
    warm_up(model_name, data, system_prompt, user_prompt, desc, settings)
    end_warmup_timestamp = datetime.now().isoformat(timespec='milliseconds')
    end_warmup_time = time.time()

    
    if batch_size == 1:
        user_prompt, system_prompt = adjust_prompt_b1(user_prompt, system_prompt)
    
    for start_i in range(0, len(data), batch_size):
        batch_segments = ",\n".join([str(i) +". "+ data[i]["segment"] for i in range(start_i, start_i+batch_size)])
        batch_id = list(range(start_i, start_i+batch_size))

        batch_prompt = user_prompt+'\n'+ batch_segments

        start_send_time = time.time()
        response = generate_response(model_name, batch_prompt, system_prompt.format(project = desc, m=start_i, n = start_i+batch_size-1), settings)
        end_send_time = time.time()

        if response["done"]:
            res_dict = {
                "batch_id": batch_id,
                "response": response["response"],
                "prompt_eval_count": response["prompt_eval_count"],
                "prompt_eval_duration": response["prompt_eval_duration"],
                "eval_count": response["eval_count"],
                "eval_duration": response["eval_duration"],
                "total_duration": response["total_duration"],
                "token_per_second": response["eval_count"] / response["total_duration"] * 1e9, 
                "created_at":  response["created_at"], 
                "start_send_time": start_send_time,
                "end_send_time": end_send_time,
                "end_time_stamp": str(datetime.now()),
                "done_reason": response["done_reason"],
                "user_prompt": batch_prompt
            }
            model_metrics["total_input_tokens"] += response["prompt_eval_count"]
            model_metrics["total_output_tokens"] += response["eval_count"]
            model_metrics["prompt_eval_duration"] += response["prompt_eval_duration"]
            model_metrics["eval_duration"] += response["eval_duration"]
            model_metrics["total_duration"] += response["total_duration"]
            model_metrics["tokens_per_sec"] += (response["eval_count"] / response["total_duration"] * 1e9)
            model_metrics["done"] += 1
            
            res_dict['system_prompt'] = system_prompt.format(project = desc, m=start_i, n = start_i+batch_size-1)
 
            
        else:
            res_dict = {
                "batch_id": batch_id, 
                "response": "",
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0,
                "total_duration": 0,
                "token_per_second": 0,
                 "created_at": 0, 
                "start_send_time": start_send_time,
                "end_send_time": end_send_time,
                "end_time_stamp": str(datetime.now()),
                "done_reason":  response["done_reason"],
                "time_stamp": str(datetime.now())
            }
        
            model_metrics["unfinished_requests"] += 1

        output_responses.append(res_dict)

    end_time = time.time()
    
    start_unload_timestamp = datetime.now().isoformat(timespec='milliseconds')
    unload_model(model_name)
    end_unload_timestamp = datetime.now().isoformat(timespec='milliseconds')
    end_unload_time = time.time()

    model_metrics["total_elapsed_time"] = end_time - end_load_time
    model_metrics["model_load_time"] = end_load_time - start_load_time
    model_metrics["model_unload_time"] = end_unload_time - end_time
    model_metrics["total_warmup_time"] = end_warmup_time - start_warmup_time
    
    model_metrics["start_load_timestamp"] = start_load_timestamp
    model_metrics["end_load_timestamp"] = end_load_timestamp
    model_metrics["start_unload_timestamp"] = start_unload_timestamp
    model_metrics["end_unload_timestamp"] = end_unload_timestamp
    model_metrics["start_warmup_timestamp"] = start_warmup_timestamp
    model_metrics["end_warmup_timestamp"] = end_warmup_timestamp


    return output_responses, model_metrics


