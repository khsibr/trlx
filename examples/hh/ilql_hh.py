import json
import os
import sys
from itertools import islice

from datasets import load_dataset
from ppo_hh import create_reward_fn
from peft import LoraConfig, TaskType

import trlx
from trlx.data.default_configs import (
    ILQLConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        batch_size=4,
        epochs=100,
        total_steps=20000,
        checkpoint_interval=10000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
        checkpoint_dir="checkpoints/ilql_hh",
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1000000000, eta_min=1e-6)),
    method=ILQLConfig(
        name="ilqlconfig",
        tau=0.6,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.0001,
        beta=0,
        steps_for_target_q_sync=1,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=128, top_k=20, beta=[1, 4], temperature=1.0),
    ),
)

config_name = os.environ.get("CONFIG_NAME")
if config_name == "125M":
    from peft import LoraConfig, TaskType

    # default_config.train.trainer_kwargs = dict(fp16=True, bf16=False,)
    default_config.method.two_qs = False
    default_config.train.batch_size = 1
    default_config.train.checkpoint_dir = "checkpoints/ilql_hh_125M"
    default_config.model.model_path = "EleutherAI/pythia-125m-deduped"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    # default_config.model.peft_config = LoraConfig(
    #     r=3,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    # )
elif config_name == "1B":
    default_config.train.batch_size = 1
    default_config.train.checkpoint_dir = "checkpoints/ilql_hh_1B"
    default_config.model.model_path = "EleutherAI/pythia-1.4b-deduped"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
elif config_name == "6B":
    default_config.method.gen_kwargs = dict(max_new_tokens=2, top_k=20, beta=[1, 4], temperature=1.0)
    default_config.train.seq_length = 10
    default_config.train.batch_size = 1
    default_config.train.checkpoint_dir = "checkpoints/ilql_hh_6B"
    default_config.model.model_path = "EleutherAI/pythia-6.9b-deduped"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.model.peft_config = LoraConfig(
        r=3,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.05,
    )
elif config_name == "20B":
    default_config.train.batch_size = 1
    default_config.train.total_steps = 3000
    default_config.train.checkpoint_dir = "checkpoints/ilql_hh_20B"
    default_config.model.model_path = "EleutherAI/gpt-neox-20b"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
elif config_name == "7B":
    default_config.method.two_qs = True
    # default_config.method.gen_kwargs = dict(max_new_tokens=2, top_k=20, beta=[1, 4], temperature=1.0)
    # default_config.train.seq_length = 10
    default_config.train.batch_size = 1
    default_config.train.checkpoint_dir = "checkpoints/Llama-2-7B-Chat-fp16-4k-sft"
    default_config.model.model_path = "ryadhkhsibfetch/Llama-2-7B-Chat-fp16-4k-sft-4"
    default_config.tokenizer.tokenizer_path = "ryadhkhsibfetch/Llama-2-7B-Chat-fp16-4k-sft-4"
    default_config.model.peft_config = LoraConfig(
        r=3,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.05,
    )

def preprocess(sample):
    sample["prompt_output"] = [
        [sample["prompt"], sample["chosen"]],
        [sample["prompt"], sample["rejected"]],
    ]
    sample["reward"] = [1, -1]
    return sample

def subsample(N, dataset, dataset_key):
    # Access the desired split
    train_dataset = dataset[dataset_key]
    # Subsample the dataset
    shuffled_dataset = train_dataset.shuffle(seed=42)
    subsampled_dataset = shuffled_dataset.select(range(N))
    # Update the DatasetDict
    dataset[dataset_key] = subsampled_dataset

def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    print(f"Train config: {config}")
    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)

    subsample(100, dataset, "train")
    subsample(100, dataset, "test")
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])
    rewards = sum(dataset["train"]["reward"], [])
    eval_prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset["test"], 280)]
    # reward_fn = create_reward_fn()

    import pickle
    with open('/home/ryadhkhsib/Dev/data/fetch/processed/rl_data.pkl', 'rb') as handle:
        rl_data = pickle.load(handle)
    prompts_outputs = rl_data["all_chats"][:10]
    rewards = rl_data["all_rewards"][:10]
    eval_prompts = [{"prompt": prompts_outputs[-i][0], "original_output": prompts_outputs[-i][1]} for i in range(10)]

    trlx.train(
        samples=prompts_outputs,
        rewards=rewards,
        config=config,
        eval_prompts=eval_prompts,
        # metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
