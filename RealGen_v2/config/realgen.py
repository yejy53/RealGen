import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    # for aigidetect model
    config.real_image_path = "" # unused
    config.d_times = 10 # unused
    config.d_lr = {"omniaid": 4e-5, "omniaid-dino": 4e-5}
    config.d_threshold = 0.6
    config.d_batch_size = 32
    config.d_target_acc = 0.90
    config.d_max_epochs = 20
    config.d_buffer_size = 1000

    config.train.preset_lora_path = "" ### for z-image-turbo
    config.train.preset_lora_weightname = ""

    config.sample.sde_type = "sde"

    config.train.vkl = False

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def flux_fast_guard():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/realgen")

    config.pretrained.model = "flux1-dev"
    config.sample.num_steps = 28
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5

    config.resolution = 1024
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(24/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 8 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.vkl = True
    config.train.beta = 4e-4
    config.train.clip_range=4e-6
    config.train.highclip_range=4e-6
    config.train.learning_rate = 1e-4
    config.rationorm = True
    config.sample.global_std = False
    config.sample.noise_level = 0.9
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)    
    config.sample.sde_type = "sde"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/realgen/flux-fast-guard'
    config.reward_fn = {
        "omniaid": 1.0,
        "omniaid-dino": 1.0,
        "visualquality": 1.0,
        "longclip": 1.0,
    }
    
    config.prompt_fn = "realgen"

    config.per_prompt_stat_tracking = True
    config.activation_checkpointing = True
    return config



def zimage_base_fast_guard():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/realgen")
    config.real_image_path = "" # In actual training, we use a paired dataset and don't use this parameter.

    config.pretrained.model = "Z-image"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0

    config.resolution = 1024
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(24/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 8 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2  # config.sample.train_batch_size * config.sample.num_batches_per_epoch // 2 // config.train.batch_size
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.vkl = True
    config.train.beta = 1e-5
    config.train.clip_range= 1e-4
    config.train.highclip_range= 1e-4
    config.train.learning_rate = 4e-4
    config.rationorm = True
    config.sample.global_std = False
    config.sample.noise_level = 0.9
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.sample.sde_type = "sde"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/realgen/z-image-base'
    config.reward_fn = {
        "omniaid": 1.0,
        "omniaid-dino": 1.0,
        "visualquality": 1.0,
        "longclip": 1.0,
    }
    
    config.prompt_fn = "realgen"

    config.per_prompt_stat_tracking = True
    config.activation_checkpointing = True
    return config


def zimage_turbo_fast_guard():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/realgen")

    # config.train.preset_lora_path = "" ### for z-image-turbo
    # config.train.preset_lora_weightname = 'zimage_turbo_training_adapter_v2.safetensors' 

    config.pretrained.model = "Z-Image-Turbo"
    config.sample.num_steps = 9
    config.sample.eval_num_steps = 9
    config.sample.guidance_scale = 0.0
    config.sample.eval_guidance_scale = 0.0

    config.resolution = 1024
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(24/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 8 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2  # config.sample.train_batch_size * config.sample.num_batches_per_epoch // 2 // config.train.batch_size
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.vkl = True
    config.train.beta = 1e-5
    config.train.clip_range= 1e-4
    config.train.highclip_range= 1e-4
    config.train.learning_rate = 4e-4
    config.rationorm = True
    config.sample.global_std = False
    config.sample.noise_level = 0.9
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2) 
    config.sample.sde_type = "sde"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/realgen/z-image-turbo'
    config.reward_fn = {
        "omniaid": 1.0,
        "omniaid-dino": 1.0,
        "visualquality": 1.0,
        "longclip": 1.0,
    }
    
    config.prompt_fn = "realgen"

    config.per_prompt_stat_tracking = True
    config.activation_checkpointing = True
    return config


def get_config(name):
    return globals()[name]()