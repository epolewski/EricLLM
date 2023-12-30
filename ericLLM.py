import sys, os, time, torch, random, asyncio, json, argparse
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_until_first_complete

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
    # Set host
    parser.add_argument('--host', metavar='HOST', type=str, default='0.0.0.0', help='Sets host')
    # Set port
    parser.add_argument('--port', metavar='PORT', type=int, default=8000, help='Sets port')
    # Set max_seq_len
    parser.add_argument('--max-model-len', metavar='MAX_SEQ_LEN', type=int, default=4096, help='Sets max_seq_len')
    # Set gpu_split
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
    parser.add_argument('--num_experts', metavar='NUM_EXPERTS', type=int, default=2,
                        help='Number of experts in a model like Mixtral (not implemented yet)')
    parser.add_argument('--cache_8bit', metavar='CACHE_8BIT', type=bool, default=False,
                        help='Use 8 bit cache (not implemented)')
    parser.add_argument('--num_workers', metavar='NUM_WORKERS', type=int, default=1,
                        help='Number of worker processes to use')



    return parser.parse_args()



args = parse_args()
print(f"Model Directory: {args.model}")
# Maximum number of generations to hold in memory before forcing a wait on new requests.
MAX_PROMPTS = args.max_prompts


app = FastAPI()

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


async def process_input(input_id):
    print()

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
            ids, response_event, max_tokens, temperature, top_k, top_p, token_repetition_penalty, stop = await prompts_queue.get()
            prompt_count += ids.size(1)
            batch_size = 1
            #if prompts_queue.qsize() % 2 == 0:
                #batch_size = 2
            cache = ExLlamaV2Cache(model, max_seq_len=(ids.size(1) + max_tokens), batch_size = batch_size)
            #print(ids)
            model.forward(ids[:, :-1], cache, preprocess_only=True)
            input_ids.append(ids)
            caches.append(cache)
            settings_clone = settings_proto.clone()
            settings_clone.temperature = temperature
            settings_clone.top_p = top_p
            settings_clone.top_k = top_k
            settings_clone.token_repetition_penalty = token_repetition_penalty

            #settings_clone.vocab_size = 8192
            #settings_clone.max_attention_size = 2048 ** 16
            #settings_clone.max_input_len = 8192
            #settings_clone.max_seq_len = 8192
            #settings_clone.hidden_size = 8192
            #settings_clone.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
            #settings.eos_token_id = 2

            #settings.append(settings_proto.clone())
            settings.append(settings_clone)
            ids_lookup[len(input_ids) - 1] = response_event #Should I change this to a hash? No because duplicates would still happen?

        token_count['read_tokens'] += prompt_count
        print(f"Doing input_ids at {time.time()}")
        while input_ids:
            inputs = torch.cat([x[:, -1:] for x in input_ids], dim=0)
            logits = model.forward(inputs, caches, input_mask=None).float().cpu()
            eos = []
            r = random.random()

            for i in range(len(input_ids)):
                token, _, _ = ExLlamaV2Sampler.sample(logits[i:i + 1, :, :], settings[i], input_ids[i], r, tokenizer)
                tempIDs = torch.cat([input_ids[i], token], dim=1)
                input_ids[i] = tempIDs

                token_count['gen_tokens'] += 1
                token_count['total_tokens'] += 1

                if token.item() == tokenizer.eos_token_id or caches[i].current_seq_len == caches[i].max_seq_len:
                    eos.insert(0, i)  # Indices of completed prompts
                    # Send the response immediately when a prompt is completed
                    output = tokenizer.decode(input_ids[i])[0].strip()
                    try:
                        response_event = ids_lookup.pop(i)
                        #input_ids.pop(i)
                        #caches.pop(i)
                        #settings.pop(i)
                    except Exception as e:
                        print(f"Error processing completed prompt: {e}")
                        continue

                    if response_event is not None:
                        data = {"text": output}
                        output = json.dumps(data, indent=2)
                        if(args.verbose == True):
                            print(output)
                        response_event.set_result(output)
                    continue

                # Remove completed prompts from the lists
            for i in eos:
                #print(f"EOS#: {i}, {eos}")
                try:
                    #ids_lookup.pop(i, None)
                    input_ids.pop(i)
                    caches.pop(i)
                    settings.pop(i)
                except:
                    print(f"Pop failed due to my crappy request lookup algorithm: {i}, {ids_lookup}")


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


@app.get('/')
def read_root():
    return {"message": "ExLlamaV2 Language Model API is running."}


@app.post('/generate', response_class=PlainTextResponse)
async def generate(prompt: PromptRequest):
    global prompts_queue, results, token_count

    encoded_prompt = tokenizer.encode(prompt.prompt)
    token_count['prompt_tokens'] += len(encoded_prompt) - 1
    completion_event = asyncio.Future()
    await prompts_queue.put((encoded_prompt, completion_event, prompt.max_tokens, prompt.temperature, prompt.top_k, prompt.top_p, prompt.token_repetition_penalty, prompt.stop))

    try:
        # Wait until the prompt is processed or timeout occurs
        return await asyncio.wait_for(completion_event, timeout=args.timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing the prompt timed out.")


def setup_model():
    global model, tokenizer
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
    #config.stop_strings = "\n"
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