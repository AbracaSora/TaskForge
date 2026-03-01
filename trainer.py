import logging
import pathlib
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")


from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from reward_model import SequenceRewardModel
from task import ClassroomSceneGenerationTask
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)




class PolicyWithValue(nn.Module):

    def __init__(self, model_name, device="cpu"):

        super().__init__()

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ===== load base =====

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu" if self.device == "cpu" else "auto",
        )

        base.gradient_checkpointing_enable()

        # ===== LoRA =====

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(
            base,
            lora_config,
        )

        self.model.print_trainable_parameters()

        hidden = self.model.config.hidden_size

        # ===== value head =====

        self.value_head = nn.Linear(
            hidden,
            1,
            bias=False,
        ).to(device).to(torch.bfloat16)

    # =====================

    def forward(self, input_ids, attention_mask):

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden = out.hidden_states[-1]

        last = hidden[:, -1]

        value = self.value_head(last)

        return out, value.squeeze(-1)

    # =====================

    def logprob_and_value(self, text):

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        out, value = self.forward(
            tokens["input_ids"],
            tokens["attention_mask"],
        )

        logits = out.logits[:, :-1]
        labels = tokens["input_ids"][:, 1:]

        # PPO 需要整段序列的 log π(a|s)，即各 token log prob 之和，不能用 mean
        loss_per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        logprob = -loss_per_token.sum()

        return logprob, value


def rollout(policy, sample):

    task = ClassroomSceneGenerationTask(
        topic=sample["topic"],
        grade=sample["grade"],
        subject=sample["subject"],
    )
    result = task.run(policy.model, policy.tokenizer)
    return result


def compute_advantage(reward, value):

    advantage = reward - value

    return advantage


def ppo_loss(new_logprob, old_logprob, adv, eps=0.2):

    ratio = torch.exp(new_logprob - old_logprob)

    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)

    loss = -torch.min(ratio * adv, clipped * adv)

    return loss.mean()


def dialogue_parser(dialogue):
    result = ""
    for turn in dialogue:
        result += f"{turn['role']}: {turn['content']}\n"
    return result






class PPOTrainer:

    def __init__(
        self,
        model_name,
        dataset,
        reward_model,
        lr=1e-6,
        device="cpu",
    ):
        self.device = device
        self.policy = PolicyWithValue(
            model_name,
            device=self.device,
        )

        self.dataset = dataset

        self.reward_model = reward_model
        self.reward_model.to(self.device)


        params = [
            p for p in self.policy.parameters()
            if p.requires_grad
        ]

        print(
            "trainable params:",
            sum(p.numel() for p in params)
        )

        self.opt = torch.optim.AdamW(
            params,
            lr=lr,
        )

    def train_step(self, sample, ppo_epochs=4, clip_eps=0.2):

        result = rollout(self.policy, sample)

        reward = torch.tensor(
            self.reward_model(result["dialogue"]),
            device=self.policy.device,
            dtype=torch.float32,
        )

        dialogue = dialogue_parser(result["full_dialogue"])

        # 关键：old_logprob 必须来自「生成该 trajectory 时的策略」且不参与梯度
        # value_ref 作为固定 baseline 用于 advantage，整段 PPO 更新内不变（与 logprob 不同，value 不做 ratio，同源无妨）
        with torch.no_grad():
            old_logprob, value_ref = self.policy.logprob_and_value(dialogue)
            old_logprob = old_logprob.detach()
            value_ref = value_ref.detach()
        adv = reward - value_ref

        total_loss_agg = 0.0
        for _ in range(ppo_epochs):
            new_logprob, value2 = self.policy.logprob_and_value(dialogue)
            policy_loss = ppo_loss(new_logprob, old_logprob, adv, eps=clip_eps)
            # value2 随参数更新而变，用于拟合 reward；与 value_ref 第一轮同源是预期行为
            value_loss = (value2 - reward).pow(2).mean()
            total_loss = policy_loss + 0.5 * value_loss

            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()
            total_loss_agg = total_loss.item()

        return total_loss_agg, reward.item()

    def train(self, steps=1000):

        for i in range(steps):
            sample = self.dataset[i % len(self.dataset)]

            loss, reward = self.train_step(sample)

            print(i, "loss", loss, "reward", reward)

if __name__ == "__main__":
    import json
    dataset = json.load(open("course.json", "r", encoding="utf-8"))
    trainer = PPOTrainer(
        model_name="Qwen/Qwen3-8B",
        dataset=Dataset.from_list(
            dataset[10:]
        ),
        reward_model=SequenceRewardModel(model="gpt-4o-mini", device="cuda"),
        device="cuda",
    )
    trainer.train(steps=100)
