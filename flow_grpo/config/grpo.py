import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/realgen")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "realgen"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def sd3_fast():
    gpu_number=7
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/realgen")

    # sd3.5 medium
    config.pretrained.model = "../RealGen/models/stable-diffusion-3.5-large"
    config.sample.num_steps = 28
    config.sample.train_num_steps = 7
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 4.5

    config.resolution = 1024
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 21
    config.sample.mini_num_image_per_prompt = 3
    config.sample.num_batches_per_epoch = int(24/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 8 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-4
    config.train.learning_rate = 1e-4
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.noise_level = 0.9
    config.train.ema = True
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/flow-grpo/sd'
    config.reward_fn = {
        "forensic_chat": 1.0,
        "omniaid": 1.0,
        "longclip": 1.0,
    }
    
    config.prompt_fn = "realgen"

    config.per_prompt_stat_tracking = True
    return config

def flux_fast():
    gpu_number=7
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/realgen")

    # sd3.5 medium
    config.pretrained.model = "../RealGen/models/flux1-dev"
    config.sample.num_steps = 28
    config.sample.train_num_steps = 7
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 1024
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 21
    config.sample.mini_num_image_per_prompt = 3
    config.sample.num_batches_per_epoch = int(24/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 8 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.noise_level = 0.9
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/flow-grpo/flux'
    config.reward_fn = {
        "forensic_chat": 1.0,
        "omniaid": 1.0,
        "longclip": 1.0,
        # "hpsv2": 1.0,
        # "hpsv3": 1.0,
        # "clipscore": 1.0,
        # "pickscore":1.0
    }
    
    config.prompt_fn = "realgen"

    config.per_prompt_stat_tracking = True
    return config

def flux_fast_guard():
    gpu_number=7
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/realgen")

    # sd3.5 medium
    config.pretrained.model = "../RealGen/models/flux1-dev"
    config.sample.num_steps = 28
    config.sample.train_num_steps = 7
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 1024
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 21
    config.sample.mini_num_image_per_prompt = 3
    config.sample.num_batches_per_epoch = int(24/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 8 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.train.clip_range=1e-5
    config.train.learning_rate = 1e-4
    config.rationorm = True
    config.sample.global_std = True
    config.sample.noise_level = 0.9
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/grpo-guard/flux-fast-guard'
    config.reward_fn = {
        "forensic_chat": 1.0,
        "omniaid": 1.0,
        "longclip": 1.0,
    }
    
    config.prompt_fn = "realgen"

    config.per_prompt_stat_tracking = True
    return config

def get_config(name):
    return globals()[name]()
