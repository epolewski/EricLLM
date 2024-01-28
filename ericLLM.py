import sys, os, time, torch, random, asyncio, json, argparse
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_until_first_complete
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming `exllamav2` can be imported with paths set correctly
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Cache_8bit,
    model_init,
    ExLlamaV2CacheBase,
    ExLlamaV2Lora,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)


def parse_args():
    parser = argparse.ArgumentParser(description="Command line arguments for a Python script.")
    # Set verbose
    parser.add_argument('--verbose', action='store_true', default=False, help='Sets verbose')
    # Set model_directory
    parser.add_argument('--model', metavar='MODEL_DIRECTORY', type=str, help='Sets model_directory')
    # Set lora
    parser.add_argument('--lora', metavar='LORA_DIRECTORY', type=str, help='Sets lora_directory')
    # Set host
    parser.add_argument('--host', metavar='HOST', type=str, default='0.0.0.0', help='Sets host')
    # Set port
    parser.add_argument('--port', metavar='PORT', type=int, default=8000, help='Sets port')
    # Set max_seq_len
    parser.add_argument('--max-model-len', metavar='MAX_SEQ_LEN', type=int, default=4096, help='Sets max_seq_len')
    # Set max_input_len
    parser.add_argument('--max-input-len', metavar='MAX_INPUT_LEN', type=int, default=4096, help='Sets max_input_len')
    # Set gpu_split
    parser.add_argument('--gpu_split', metavar='GPU_SPLIT', type=str, default='',
                        help='Sets array gpu_split and accepts input like 16,24')
    # Set gpu_balance
    parser.add_argument('--gpu_balance', action='store_true', default=False,
                        help='Balance workers on GPUs to maximize throughput. Make sure --gpu_split is set to the full memory of all cards.')
    # Set MAX_PROMPTS
    parser.add_argument('--max_prompts', metavar='MAX_PROMPTS', type=int, default=16, help='Max prompts to process at once')
    # Set timeout
    parser.add_argument('--timeout', metavar='TIMEOUT', type=float, default=30.0, help='Sets timeout')
    # Set alpha_value
    parser.add_argument('--alpha_value', metavar='ALPHA_VALUE', type=float, default=1.0, help='Sets alpha_value')
    # Set compress_pos_emb
    parser.add_argument('--compress_pos_emb', metavar='COMPRESS_POS_EMB', type=float, default=1.0,
                        help='Sets compress_pos_emb')
    parser.add_argument('--embiggen', metavar='embiggen', type=int, default=0,
                        help='Duplicates some attention layers this many times to make larger frankenmodels dynamically. May increase cromulence on benchmarks.')
    parser.add_argument('--num_experts', metavar='NUM_EXPERTS', type=int, default=2,
                        help='Number of experts in a model like Mixtral (not implemented yet)')
    parser.add_argument('--cache_8bit', metavar='CACHE_8BIT', type=bool, default=False,
                        help='Use 8 bit cache (not implemented)')
    parser.add_argument('--num_workers', metavar='NUM_WORKERS', type=int, default=1,
                        help='Number of worker processes to use')
     # Add a new command-line option for cloud management
    default_config_api_url = 'https://default-config-api.com/config'
    parser.add_argument('--managed', metavar='MANAGED', type=str, default=default_config_api_url,
                        help=f'Enables online management of this server. (default: {default_config_api_url})')
    default_config_engine = "ExLlamaV2"
    parser.add_argument('--engine', metavar='ENGINE', type=str, default=default_config_engine,
                        help=f'Choose the underlying engine of this server. Supports vLLM and ExLlamaV2 in quotes (default: {default_config_engine})')



    return parser.parse_args()



args = parse_args()
print(f"Model Directory: {args.model}")
# Maximum number of generations to hold in memory before forcing a wait on new requests.
MAX_PROMPTS = args.max_prompts


app = FastAPI()

if(args.engine == "vLLM"):
    from vllm import LLM, SamplingParams


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.5
    token_repetition_penalty: float = 1.00
    stop: list = []
    skip_special_tokens: bool = True
    n: int = 1

# Globals to store states
prompts_queue = asyncio.Queue()
results = {}
token_count = {'prompt_tokens': 0, 'gen_tokens': 0, 'read_tokens' : 0, 'total_tokens' : 0}
processing_started = False
model = None
tokenizer = None
total_processing_time = 0
loras = []


async def process_input(input_id):
    print()

try:
    async def inference_loop():
        global prompts_queue, results, token_count, processing_started, total_processing_time
        processing_started = True
        token_processing_start_time = None


        while processing_started:
            # Quick sleep to let the API server send back requests, or it waits serially for all these to finish
            await asyncio.sleep(0.01)
            if prompts_queue.qsize() == 0:
                await asyncio.sleep(0.1)
                continue  # No prompts to process yet.

            # Start timing the token processing
            #if token_processing_start_time is None:
            token_processing_start_time = time.time()
            token_count['read_tokens'] = 0
            token_count['prompt_tokens'] = 0
            token_count['gen_tokens'] = 0

            input_ids = []
            caches = []
            settings = []

            ids_lookup = {}
            settings_proto = ExLlamaV2Sampler.Settings()

            prompt_count = 0
            print(f"Starting at {time.time()}")
            for _ in range(min(MAX_PROMPTS, prompts_queue.qsize())):

                ids, response_event, max_tokens, temperature, top_k, top_p, token_repetition_penalty, stop, prompt = await prompts_queue.get()
                #print(prompt)
                if(args.engine == "vLLM"):
                    sampling_params = SamplingParams(temperature=temperature, top_p=top_p,max_tokens=max_tokens, top_k=top_k)
                    outputs = model.generate(prompt,sampling_params)
                    #print(f"Outputs {outputs}")
                    #print(f"Outputs[0] {outputs[0]}")

                    #print(f"outputs.outputs {outputs[0].outputs[0]}")
                    #
                    for output in outputs:
                        data = {"text": output.outputs[0].text}
                        jsond = json.dumps(data, indent=2)
                        response_event.set_result(jsond)
                    continue

                prompt_count += ids.size(1)
                batch_size = 1
                if(args.cache_8bit):
                    cache = ExLlamaV2Cache_8bit(model, max_seq_len=(ids.size(1) + max_tokens), batch_size = batch_size)
                else:
                    cache = ExLlamaV2Cache(model, max_seq_len=(ids.size(1) + max_tokens), batch_size = batch_size)


                if(args.lora):
                    model.forward(ids[:, :-1], cache, preprocess_only=True, loras = loras)
                else:
                    model.forward(ids[:, :-1], cache, preprocess_only=True)
                input_ids.append(ids)
                caches.append(cache)
                settings_clone = settings_proto.clone()
                settings_clone.temperature = temperature
                settings_clone.top_p = top_p
                settings_clone.top_k = top_k
                settings_clone.token_repetition_penalty = token_repetition_penalty
                settings_clone.batch_size = 1
                #settings_clone.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
                settings_clone.eos_token_id = 2
                settings_clone.eos_token = "</s>"

                #settings.append(settings_proto.clone())
                settings.append(settings_clone)
                ids_lookup[len(input_ids) - 1] = response_event #Should I change this to a hash? No because duplicates would still happen?
            
            # Just skip all this since it doesn't work if using the vllm engine

            if(args.engine == "vLLM"):
                continue

            token_count['read_tokens'] += prompt_count
            print(f"Doing input_ids at {time.time()}")
            while input_ids:
                inputs = torch.cat([x[:, -1:] for x in input_ids], dim=0)
                if(args.lora):
                    logits = model.forward(inputs, caches, input_mask=None, loras = loras).float().cpu()
                else:
                    logits = model.forward(inputs, caches, input_mask=None).float().cpu()
                eos = []
                r = random.random()

                for i in range(len(input_ids)):
                    token, _, _ = ExLlamaV2Sampler.sample(logits[i:i + 1, :, :], settings[i], input_ids[i], r, tokenizer)
                    tempIDs = torch.cat([input_ids[i], token], dim=1)
                    input_ids[i] = tempIDs

                    token_count['gen_tokens'] += 1
                    token_count['total_tokens'] += 1
                    #stop_token = tokenizer.encode("</s>")
                    if token.item() == tokenizer.eos_token_id or caches[i].current_seq_len == caches[i].max_seq_len:
                        if token.item() == settings[i].eos_token_id:
                            print(f"Stopping for token: {token.item()}, settings eos: {settings[i].eos_token_id}, tokenizer eos: {tokenizer.eos_token_id}")
                        eos.insert(0, i)  # Indices of completed prompts
                        # Send the response immediately when a prompt is completed
                        output = tokenizer.decode(input_ids[i])[0].strip()
                        try:
                            if i in ids_lookup:
                                response_event = ids_lookup.pop(i, None)
                                if response_event is not None:
                                    data = {"text": output}
                                    output = json.dumps(data, indent=2)
                                    if (args.verbose == True):
                                        print(output)
                                    response_event.set_result(output)
                                else:
                                    print(f"Response event is None")
                            else:
                                print(f"ID was not in ids_lookup")
                        except Exception as e:
                            print(f"Error processing completed prompt: {e}")
                            continue


                        #
                        continue

                    # Remove completed prompts from the lists

                for i in eos:
                    try:
                        #ids_lookup.pop(i, None)
                        input_ids.pop(i)
                        caches.pop(i)
                        settings.pop(i)
                    except:
                        print(f"Pop failed due to my crappy request lookup algorithm: {i}, {ids_lookup}")
            #print(f"IDs lookup left over: {ids_lookup}")
            try:
                for i in ids_lookup:
                    print(f"Jobs came back late, recovering: {ids_lookup}")
                    response_event = ids_lookup.pop(i, None)
                    if response_event is not None:
                        data = {"text": output}
                        output = json.dumps(data, indent=2)
                        if (args.verbose == True):
                            print(output)
                        response_event.set_result(output)
                    else:
                        print(f"Response event is None")
            except Exception as e:
                print(f"Error processing completed prompt: {e}")
                continue


            current_time = time.time()
            time_elapsed_seconds = current_time - token_processing_start_time
            total_processing_time += time_elapsed_seconds
            read_speed = token_count['read_tokens'] / time_elapsed_seconds
            generation_speed = token_count['gen_tokens'] / time_elapsed_seconds
            average_gen_speed =  token_count['total_tokens'] / total_processing_time

            # Log stats to the console
            print(f"Batch process done. Read {token_count['read_tokens']} tokens at {read_speed:.2f} tokens/s. "
                  f"Generated {token_count['gen_tokens']} tokens at {generation_speed:.2f} tokens/s.\n"
                  f"This thread generated a total of {token_count['total_tokens']} tokens at {average_gen_speed:.2f} tokens/s.")
            token_processing_start_time = None  # Reset the start time
except:
    print("Model processing died. Attempting reload.")
    setup_model()
    asyncio.create_task(inference_loop())


@app.get('/')
def read_root():
    return {"message": "ExLlamaV2 Language Model API is running."}


@app.post('/generate', response_class=PlainTextResponse)
async def generate(prompt: PromptRequest):
    global prompts_queue, results, token_count
    if(args.engine == "vLLM"):
        encoded_prompt = prompt.prompt
        token_count['prompt_tokens'] += len(encoded_prompt) - 1
    else:
        encoded_prompt = tokenizer.encode(prompt.prompt)
        token_count['prompt_tokens'] += len(encoded_prompt) - 1

    completion_event = asyncio.Future()
    await prompts_queue.put((encoded_prompt, completion_event, prompt.max_tokens, prompt.temperature, prompt.top_k, prompt.top_p, prompt.token_repetition_penalty, prompt.stop, prompt.prompt))

    try:
        # Wait until the prompt is processed or timeout occurs
        if(args.timeout):
            return await asyncio.wait_for(completion_event, timeout=args.timeout)
        else:
            return await asyncio.wait_for(completion_event)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing the prompt timed out.")


def setup_model():
    global model, tokenizer, loras
    
    if(args.engine == "vLLM"):
        import ray, torch
        ray.shutdown()
        ray.init(num_gpus=torch.cuda.device_count())
        if(args.gpu_split):
            count = len(args.gpu_split.split(","))
            model = LLM(model=args.model,max_context_len_to_capture=args.max_model_len,tensor_parallel_size=count)
        else:
            model = LLM(model=args.model,max_context_len_to_capture=args.max_model_len)
        return

    model_directory = args.model
    config = ExLlamaV2Config()
    config.model_dir = model_directory
    config.prepare()
    config.scale_pos_emb = args.compress_pos_emb
    config.scale_alpha_value = args.alpha_value
    config.max_seq_len = args.max_model_len
    config.max_input_len = args.max_input_len
    #config.num_experts_per_token = 2
    #config.num_experts_per_tok = 2
    #config.num_experts = args.num_experts
    #config.num_key_value_heads = args.num_experts
    #config.num_local_experts = 2
    #config.q_handle = 2
    config.max_batch_size = 1
    #config.filters = "</s>"
    config.stop_strings = "</s>"
    config.eos_token_id = 2
    #config.qkv_embed = True


    print("Loading model: " + model_directory)
    model = ExLlamaV2(config)
    if(args.gpu_split):
        sleep_time = random.uniform(0.1, 3)
        time.sleep(sleep_time)
        if (args.gpu_balance):
            while(os.path.exists("gpu_assign.lock")):
                time.sleep(0.3)
            with open("gpu_assign.lock", 'w', encoding='utf-8') as file:
                file.write("")
            # Read the first line, remove it, and write the rest back to the file
            with open("gpu_assign", 'r+', encoding='utf-8') as file:
                # Read the first line
                first_line = file.readline().replace("\n","")

                # Read the rest of the file
                rest_of_content = file.read()

                # Move the cursor to the beginning of the file
                file.seek(0)

                # Write the rest of the content back to the file
                file.write(rest_of_content)

                # Truncate the file to remove any remaining characters from the old content
                file.truncate()
                print(first_line)
            try:
                os.remove("gpu_assign.lock")
            except OSError as e:
                print(f"Error removing lock: {e}")

            gpus = list(map(int, first_line.split(',')))

        else:
            gpus = list(map(int, args.gpu_split.split(',')))
        model.load(gpu_split=gpus)
    else:
        model.load()
    tokenizer = ExLlamaV2Tokenizer(config)
    print("Model is loaded.")
    if(args.lora):
        lora = ExLlamaV2Lora.from_directory(model, args.lora)
        loras.append(lora)

    # Embiggen the model x times without increasing memory usage
    for i in range(args.embiggen):
        ## mix layers here
        layer_arrangement = list(range(0, 14)) + list(range(4, 22))
        #list(range(8, 18)) +
        # modules arangement: [embedding, [...layers], rms-norm, head]
        # where each layer is [attention, mlp]
        old_modules = model.modules
        model.modules = old_modules[:1]
        for idx in layer_arrangement:
            model.modules += old_modules[idx * 2 + 1: idx * 2 + 3]
        model.modules += old_modules[-2:]
        model.head_layer_idx = len(model.modules) - 1
        model.config.num_hidden_layers = len(layer_arrangement)
        model.last_kv_layer_idx = len(model.modules) - 4


@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    setup_model()
    asyncio.create_task(inference_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global processing_started
    processing_started = False
    print("Shutting down...")


if __name__ == "__main__":
    import uvicorn

    # Clean up any previous file locks
    if(os.path.exists("gpu_assign")):
        print(f"Deleting old gpu assignment file")
        os.remove("gpu_assign")
    if(os.path.exists("gpu_assign.lock")):
        print(f"Deleting old gpu lock file")
        os.remove("gpu_assign.lock")


    # global worker_assignments
    # worker_assignments = []
    # Load balance workers across GPUs
    if (args.gpu_balance):
        gpus = list(map(int, args.gpu_split.split(',')))
        average_workers = int(args.num_workers / len(gpus))
        content = ""
        for i in range(args.num_workers):
            gpu_mapping = []
            for j in range(len(gpus)):
                # If the number of workers doesn't fit evenly on the cards, distribute the odd ones out. Since exllamav2 doesn't
                # distribute perfectly with --gpu_split, I'm going to just guess at it now with a formula. There's probably a more
                # clever way to split them up perfectly, I just haven't come up with it yet.
                if ((i + 1 + args.num_workers % len(gpus) > args.num_workers) and args.num_workers % len(gpus) != 0):
                    #if i % len(gpus) != j:
                    gpu_mapping.append(int(gpus[j] / len(gpus) + 2))
                    #else:
                        #gpu_mapping.append(gpus[j])
                else:
                    if i % len(gpus) == j:
                        gpu_mapping.append(gpus[j])
                    else:
                        gpu_mapping.append(0)
            text_mapping = ','.join(map(str, gpu_mapping))
            content += text_mapping + "\n"
        with open("gpu_assign", 'w', encoding='utf-8') as file:
            file.write(content)

    print(f"Starting a server at {args.host} on port {args.port}...")
    uvicorn.run("__main__:app", host=args.host, port=args.port, workers = args.num_workers, http="h11")