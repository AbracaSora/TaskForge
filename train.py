"""
本地大模型训练脚本：SFT（监督微调）+ 强化学习（PPO）
支持 Huggingface 模型 ID 或本地路径。依赖：transformers peft trl datasets accelerate bitsandbytes
"""

import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from reward_model import ExampleRewardModel, SequenceRewardModel
from task import ClassroomSceneGenerationTask


def messages_to_prompt(messages):
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ============== 参数 ==============
model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"  # 或本地路径如 /path/to/model
stage = "both"  # "sft" | "rl" | "both"
output_dir = "./output"
sft_output_dir = None  # 默认 output_dir/sft
rl_output_dir = None  # 默认 output_dir/rl

# ============== 流程 ==============
os.makedirs(output_dir, exist_ok=True)
sft_dir = sft_output_dir or os.path.join(output_dir, "sft")
rl_dir = rl_output_dir or os.path.join(output_dir, "rl")

print("加载模型与分词器:", model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    ),
)
model.print_trainable_parameters()

# ---------- SFT ----------
if stage in ("sft", "both"):
    print("=" * 50, "阶段一：SFT", "=" * 50)
    os.makedirs(sft_dir, exist_ok=True)

    ds = Dataset.from_list(
        [
            {
                "text": "Instruction: 请用一句话介绍 Python。\nResponse: Python 是一种广泛使用的高级编程语言，以简洁易读著称。"
            },
            {"text": "Instruction: 1+1 等于几？\nResponse: 1+1 等于 2。"},
        ]
    )

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=sft_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        ),
        train_dataset=ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )
    trainer.train()
    trainer.save_model(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    print("SFT 模型已保存至", sft_dir)

# ---------- RL (PPO) ----------
if stage in ("rl", "both"):
    print("=" * 50, "阶段二：强化学习 (PPO)", "=" * 50)
    os.makedirs(rl_dir, exist_ok=True)
    rl_model_path = sft_dir if stage == "both" else model_name_or_path

    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(rl_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(rl_model_path)
    tok = AutoTokenizer.from_pretrained(rl_model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ppo_config = PPOConfig(
        batch_size=1,
        learning_rate=1.41e-5,
        log_with=None,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config, model=ppo_model, ref_model=ref_model, tokenizer=tok
    )

    reward_model = SequenceRewardModel()
    course_path = Path(__file__).resolve().parent / "course.json"
    with open(course_path, "r", encoding="utf-8") as f:
        courses = json.load(f)

    num_turns = 50
    max_new_tokens_per_turn = 1024
    rl_epochs = 10

    for epoch in range(rl_epochs):
        course = random.choice(courses)
        task = ClassroomSceneGenerationTask(
            topic=course["topic"],
            subject=course["subject"],
            grade=course["grade"],
            max_turns=num_turns,
            max_tokens=max_new_tokens_per_turn,
            temperature=0.2,
        )
        dialogue = []
        all_query_tensors = []
        all_response_tensors = []
        replies = []

        for t in range(num_turns):
            if t == 0:
                prompt_dict = task._build_first_utterance_prompt(
                    task.topic, task.subject, task.grade
                )
            else:
                prompt_dict = task._build_next_utterance_prompt(
                    dialogue,
                    topic=task.topic,
                    subject=task.subject,
                    grade=task.grade,
                )
            prompt_text = messages_to_prompt(prompt_dict["messages"])
            q = tok(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            query_t = q["input_ids"][0]
            all_query_tensors.append(query_t)

            resp_tensors = ppo_trainer.generate(
                [query_t],
                max_new_tokens=max_new_tokens_per_turn,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
            )
            one_resp = resp_tensors[0]
            all_response_tensors.append(one_resp)
            one_resp_text = tok.decode(one_resp.squeeze(), skip_special_tokens=True)
            utterance = task._parse_utterance(one_resp_text, default_speaker="学生")
            dialogue.append(utterance)
            replies.append(utterance)

            if "下课" in utterance or "再见" in utterance:
                break

        reward = reward_model.score(replies)
        reward_t = torch.tensor(reward, dtype=torch.float)
        n = len(all_query_tensors)
        for k in range(n):
            stats = ppo_trainer.step(
                [all_query_tensors[k]], [all_response_tensors[k]], [reward_t]
            )
            if stats:
                print(
                    "RL Epoch",
                    epoch,
                    "course",
                    course.get("id"),
                    course.get("topic"),
                    "turn",
                    k,
                    "approx_kl:",
                    stats.get("objective/kl", 0),
                    "reward:",
                    reward_t.item(),
                )

    ppo_model.save_pretrained(rl_dir)
    tok.save_pretrained(rl_dir)
    print("RL 模型已保存至", rl_dir)

print("训练完成.")
