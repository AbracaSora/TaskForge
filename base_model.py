"""
BaseModel 基类：统一初始化（本地/HF）、SFT/RL 抽象接口、保存。
子类需实现：sft_config / sft_train、rl_config / rl_train。
"""

import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from reward_model import RewardModelBase


class BaseModel(ABC):
    """
    基类：支持从本地目录或 Hugging Face 加载；SFT/RL 各两个抽象方法（config + train）；保存到指定路径。
    """

    def __init__(self, model_name_or_path: str, use_4bit: bool = False):
        """
        读取本地或 HF 模型。
        Args:
            model_name_or_path: 本地目录路径或 Hugging Face 模型 ID。
            use_4bit: 是否使用 4bit 量化。
        """
        self.model_name_or_path = model_name_or_path
        self.use_4bit = use_4bit
        self._sft_config: Optional[Any] = None
        self._rl_config: Optional[Any] = None
        self._model, self._tokenizer = self._load_model_and_tokenizer(
            model_name_or_path, use_4bit
        )

    def _load_model_and_tokenizer(self, path: str, use_4bit: bool):
        is_local_dir = os.path.exists(path) and os.path.isdir(path)
        if is_local_dir:
            path = os.path.abspath(path)

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else None
        model_kwargs = dict(
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            trust_remote_code=True,
            device_map=device_map,
        )
        if use_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        if is_local_dir:
            adapter_config = os.path.join(path, "adapter_config.json")
            if os.path.isfile(adapter_config):
                import json

                with open(adapter_config, "r") as f:
                    cfg = json.load(f)
                base_name = cfg.get("base_model_name_or_path", path)
                model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
                model = PeftModel.from_pretrained(model, path)
                model = model.merge_and_unload()
            else:
                model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)

        if not torch.cuda.is_available():
            model = model.to("cpu")
        model.eval()
        return model, tokenizer

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    # ---------- SFT ----------
    @abstractmethod
    def sft_config(self, config: Any) -> None:
        """接收 SFT 配置并保存，供 sft_train 使用。"""
        pass

    @abstractmethod
    def sft_train(self, dataset: Any) -> None:
        """接收 dataset，使用已设置的 SFT config 开始训练。"""
        pass

    # ---------- RL ----------
    @abstractmethod
    def rl_config(self, config: Any) -> None:
        """接收 RL 配置并保存，供 rl_train 使用。"""
        pass

    @abstractmethod
    def rl_train(self, reward_model: RewardModelBase) -> None:
        """接收 RewardModel，使用已设置的 RL config 开始训练。"""
        pass

    # ---------- 保存 ----------
    def save_pretrained(self, path: str) -> None:
        """保存模型与分词器至指定路径。"""
        os.makedirs(path, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)


class SequentialLLM(BaseModel):
    # ---------- SFT ----------
    def sft_config(self, config: Any) -> None:
        self._sft_config = config

    def sft_train(self, dataset: Any) -> None:
        if self._sft_config is None:
            raise ValueError("SFT config not set")

        training_args = TrainingArguments(**self._sft_config)

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self._tokenizer,
        )

        trainer.train()

        # 更新模型（Trainer 内部会更新参数）
        self._model = trainer.model
        self._model.eval()

    # ---------- RL ----------
    def rl_config(self, config: Any) -> None:
        self._rl_config = config

    def rl_train(self, reward_model) -> None:
        if self._rl_config is None:
            raise ValueError("RL config not set")

        # PPO 需要 value head
        from trl import AutoModelForCausalLMWithValueHead

        policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name_or_path
        )

        # 把 SFT 权重加载进 policy_model
        policy_model.pretrained_model.load_state_dict(
            self._model.state_dict(), strict=False
        )

        ppo_config = PPOConfig(**self._rl_config)

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=policy_model,
            tokenizer=self._tokenizer,
        )

        # 伪代码：你自己定义 rollout 逻辑
        for batch in ...:
            queries = batch["query"]
            responses = ppo_trainer.generate(queries)

            rewards = reward_model.compute(queries, responses)

            ppo_trainer.step(queries, responses, rewards)

        # 训练结束后更新 self._model
        self._model = policy_model.pretrained_model
        self._model.eval()
