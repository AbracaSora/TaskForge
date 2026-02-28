"""
Task 基类：输入一个模型，经过设计好的任务逻辑，返回 response（类型任意）。
提供 _invoke(model, tokenizer, prompt_or_messages) 模仿 OpenAI API 的输入输出。

invoke 支持 JSON 格式输入，与 OpenAI Chat Completions 一致，例如：
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "max_tokens": 1024, "temperature": 0.2}
解析后会将 system 拼入 prompt（System -> User -> Assistant），再调用模型。
"""

from pathlib import Path
from dotenv import load_dotenv

from reward_model import SequenceRewardModel

load_dotenv(Path(__file__).resolve().parent / ".env")
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json


class Task(ABC):
    """
    任务基类：子类实现 run(model, tokenizer)；invoke 提供 OpenAI 风格的对话调用。

    若返回值要直接用于 ppo_trainer.step，run 应返回包含以下键的 dict：
      - query_tensors: List[torch.Tensor]，每轮 prompt 的 input_ids，每元素 shape (seq_len,)
      - response_tensors: List[torch.Tensor]，每轮「仅生成部分」的 token ids，每元素 shape (resp_len,)
      - reward: float 或 torch.Tensor 标量，整段对话的奖励
    调用方即可循环：step([query_tensors[k]], [response_tensors[k]], [reward])
    """

    def __init__(self):
        self.full_dialogue = []

    @abstractmethod
    def run(self, model: Any, tokenizer: Any) -> Any:
        """
        输入模型，执行任务，返回 response。

        Args:
            model: 任意可用的模型对象（如 BaseModel 实例、API 客户端等），由子类约定接口。
        Returns:
            response: 任务输出，类型由子类决定（字符串、列表、字典等）。
        """
        pass

    def _parse_messages_to_prompt(
        self,
        prompt: Union[str, dict, list],
        tokenizer: Any,
    ) -> dict:
        """将 OpenAI messages 转为模型输入的 prompt 字符串。"""
        messages = None
        if isinstance(prompt, str):
            s = prompt.strip()
            if s.startswith("{"):
                try:
                    data = json.loads(s)
                    messages = data.get("messages")
                except json.JSONDecodeError:
                    messages = [{"role": "user", "content": prompt}]
            else:
                # 纯字符串视为单条 user 消息
                messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            messages = prompt.get("messages")
        elif isinstance(prompt, list):
            messages = prompt

        self.full_dialogue.extend(messages)

        if not messages:
            raise ValueError("invoke: 无法从输入中解析出有效的 messages 或 prompt。")

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            print("TypeError: enable_thinking=True is not supported")
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return {
            "text": text,
        }

    def _invoke(
        self,
        model: Any,
        tokenizer: Any,
        prompt: Union[str, dict, list],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        model_id: str = "local",
    ) -> Any:
        """
        输入 OpenAI 风格的 JSON（或兼容格式），调用模型生成，返回模仿 OpenAI Chat Completions 的响应结构。
        支持三种输入形式：
        1. dict：如 {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "max_tokens": 1024, "temperature": 0.2}
        2. JSON 字符串：同上，可被 json.loads 解析
        3. 仅 messages 列表：直接 list of message dict
        解析后会将 system 等消息拼入 prompt（System -> User -> Assistant -> ...），再调用模型。

        Args:
            model: 具备 .tokenizer 与 .model 的模型（或子类重写后可为 API 客户端）。
            prompt_or_messages: 输入：dict/JSON 字符串（含 messages），或 messages 列表；也兼容单字符串作为 user 内容。
            max_tokens: 最大生成长度（可被 JSON 内同名参数覆盖）。
            temperature: 采样温度（可被 JSON 内同名参数覆盖）。
            model_id: 响应中的 model 字段。

        Returns:
            形如 OpenAI 的 dict：id, object, created, model, choices[{ message: { role, content }, finish_reason }], usage。
        """
        # 解析 JSON 或兼容格式，得到 messages 与可选参数
        tok = tokenizer
        m = model
        text = self._parse_messages_to_prompt(prompt, tokenizer)["text"]

        model_inputs = tok([text], return_tensors="pt").to("cuda")
        prompt_tokens = model_inputs["input_ids"].shape[-1]
        with torch.no_grad():
            generated_ids = m.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        output_ids = generated_ids[0][prompt_tokens:].tolist()
        completion_tokens = len(output_ids)
        # 解析 thinking：151668 为 </think>，其后为最终回复
        try:
            idx = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            idx = 0
        content = tok.decode(output_ids[idx:], skip_special_tokens=True).strip()

        self.full_dialogue.append({"role": "assistant", "content": content})

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


# 课堂场景生成（模仿 Agent.py 的提示与流程，用 _invoke 调用模型）
CLASSROOM_STYLE_GUIDE = """
【开场】真实课堂常以「老师：上课。」「学生：起立，老师好。」开场；随后老师可问预习、引入课题等。
【发言风格】学生可极简短（如「对。」「1。」「不全面。」）；老师可点名「你来说」「谁来补充」；内容需贴合当前科目与课堂主题。
【结束】课堂结束时常为老师布置作业或说「下课」、学生「起立，老师再见」。

<角色>:<内容>视为一条发言，不要有任何分析、说明或复述题目。
输出要求：
- 只输出一行，格式为：老师：<内容> 或 学生：<内容>，由上下文决定谁说话。
- 学生可极简短（如「对。」「好的。」）；老师可点名；内容贴合科目与主题。
- 禁止输出「老师：发言内容」「学生：发言内容」等占位符；禁止输出「好的，我需要分析…」「根据当前对话，生成…」等。直接以「老师：」或「学生：」开头。
"""


class ClassroomSceneGenerationTask(Task):
    """
    课堂场景生成任务：用提示词引导第一句发言，再根据上下文逐句生成下一句，
    返回完整对话列表。逻辑模仿 Agent.py，通过 _invoke(model, prompt) 调用模型。
    """

    def __init__(
        self,
        topic: str = "我与地坛",
        subject: str = "语文",
        grade: str = "高中一年级",
        max_turns: int = 50,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.topic = topic
        self.subject = subject
        self.grade = grade
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reward_model = SequenceRewardModel()

    def run(self, model: Any, tokenizer: Any) -> list:
        """
        执行一轮课堂对话生成：首句 + 循环「根据上下文生成下一句」。
        返回 {"dialogue": ["老师：…", "学生：…", ...], "topic": ..., "subject": ..., "grade": ...}。
        """
        dialogue = []

        # 1. 第一句发言
        first_prompt = self._build_first_utterance_prompt(
            self.topic, self.subject, self.grade
        )
        first_resp = self._invoke(
            model,
            tokenizer,
            first_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        first_content = (
            first_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        print(first_content)
        print("--------------------------------")
        first_utterance = self._parse_utterance(first_content, default_speaker="老师")
        if first_utterance.startswith("学生："):
            first_utterance = "老师：" + first_utterance[3:]
        if not first_utterance.startswith("老师："):
            first_utterance = (
                f"老师：{first_utterance}" if first_utterance else "老师：上课。"
            )
        dialogue.append(first_utterance)

        # 2. 逐句生成下一句
        for _ in range(self.max_turns - 1):
            next_prompt = self._build_next_utterance_prompt(
                dialogue,
                topic=self.topic,
                subject=self.subject,
                grade=self.grade,
            )
            next_resp = self._invoke(
                model,
                tokenizer,
                next_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            next_content = (
                next_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            print(next_content)
            print("--------------------------------")
            next_utterance = self._parse_utterance(next_content)
            dialogue.append(next_utterance)
            if "下课" in next_utterance or "再见" in next_utterance:
                break

        initial_prompt = self._parse_messages_to_prompt(
            self.full_dialogue[:1], tokenizer
        )["text"]
        query_t = tokenizer(initial_prompt, return_tensors="pt").to("cuda")
        response = self._parse_messages_to_prompt(self.full_dialogue[1:], tokenizer)[
            "text"
        ]
        response_t = tokenizer(response, return_tensors="pt").to("cuda")
        reward = self.reward_model.score(dialogue)

        return {
            "dialogue": dialogue,
            "topic": self.topic,
            "subject": self.subject,
            "grade": self.grade,
            "query_tensors": query_t,
            "response_tensors": response_t,
            "reward": reward,
        }

    def _build_first_utterance_prompt(
        self, topic: str, subject: str, grade: str
    ) -> str:
        prompt = f"""你正在模拟一堂课的对话。根据以下信息，直接输出老师的**第一句发言**，不要有任何分析、说明或复述题目。

课堂主题：{topic}；科目：{subject}；年级：{grade}

输出要求：
- 只输出一行，格式为：老师：<真实发言内容>
- 第一句可为「老师：上课。」或「老师：同学们，今天我们学习……」
- 禁止输出「老师：发言内容」等占位符；禁止输出「好的，我需要…」「根据当前对话…」等分析语。直接以「老师：」开头。"""
        return {
            "messages": [
                {"role": "system", "content": CLASSROOM_STYLE_GUIDE},
                {"role": "user", "content": prompt},
            ],
        }

    def _build_next_utterance_prompt(
        self,
        dialogue_so_far: list,
        topic: str = "",
        subject: str = "",
        grade: str = "",
    ) -> str:
        ctx = "\n".join(dialogue_so_far[-12:]) if dialogue_so_far else ""
        course_line = (
            f"\n当前课堂：{' '.join(p for p in [subject, grade, topic] if p)}\n"
            if (subject or grade or topic)
            else ""
        )
        prompt = f"""根据下面已有对话，直接输出**下一条**发言（仅老师或学生一方的一条），不要有任何分析、说明或复述题目。
{course_line}
已有对话：
{ctx}
输出要求：
- 只输出一行，格式为：老师：<内容> 或 学生：<内容>，由上下文决定谁说话。
- 学生可极简短（如「对。」「好的。」）；老师可点名；内容贴合科目与主题。
- 禁止输出「老师：发言内容」「学生：发言内容」等占位符；禁止输出「好的，我需要分析…」「根据当前对话，生成…」等。直接以「老师：」或「学生：」开头。"""
        return {
            "messages": [
                {"role": "system", "content": CLASSROOM_STYLE_GUIDE},
                {"role": "user", "content": prompt},
            ],
        }

    def _parse_utterance(self, text: str, default_speaker: str = "学生") -> str:
        """从模型输出中解析出单条发言，格式为 老师：xxx 或 学生：xxx。"""
        text = (text or "").strip()
        if "老师：" in text:
            line = text.split("老师：")[-1].split("学生：")[0].strip()
            return f"老师：{line}" if line else "老师：请继续。"
        if "学生：" in text:
            line = text.split("学生：")[-1].split("老师：")[0].strip()
            return f"学生：{line}" if line else "学生：好的。"
        if text:
            return f"{default_speaker}：{text[:80]}"
        return "学生：好的。" if default_speaker == "学生" else "老师：请继续。"


def load_courses(course_path: Union[str, Path], course_ids: List[int] = None) -> List[dict]:
    """从 JSON 文件加载课程列表，可选按 id 过滤。"""
    path = Path(course_path)
    if not path.exists():
        raise FileNotFoundError(f"课程文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        courses = json.load(f)
    if course_ids is not None:
        id_set = set(course_ids)
        courses = [c for c in courses if c.get("id") in id_set]
    return courses


def _parse_dialogue_from_text(text: str) -> List[str]:
    """从整段文本解析出 老师/学生 发言列表，供 SequenceRewardModel 使用。"""
    out = []
    text = (text or "").strip()
    for part in re.split(r"(?=老师：|学生：)", text):
        part = part.strip()
        if part.startswith("老师：") or part.startswith("学生："):
            out.append(part[:80] if len(part) > 80 else part)
    return out if out else ["老师：" + text[:80] if text else "老师：上课。"]


def _build_ppo_dataset(courses: List[dict], tokenizer: Any, style_guide: str) -> "Dataset":
    """为 experimental PPO 构建 Dataset：每门课一条，input_ids 为第一句发言的 prompt。"""
    from datasets import Dataset

    rows = []
    for c in courses:
        topic = c.get("topic", "")
        subject = c.get("subject", "")
        grade = c.get("grade", "")
        prompt_dict = {
            "messages": [
                {"role": "system", "content": style_guide},
                {
                    "role": "user",
                    "content": f"""你正在模拟一堂课的对话。根据以下信息，直接输出老师的**第一句发言**，不要有任何分析、说明或复述题目。
课堂主题：{topic}；科目：{subject}；年级：{grade}
输出要求：只输出一行，格式为：老师：<真实发言内容>。直接以「老师：」开头。""",
                },
            ],
        }
        try:
            text = tokenizer.apply_chat_template(
                prompt_dict["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                prompt_dict["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        rows.append({
            "input_ids": enc["input_ids"][0].tolist(),
            "attention_mask": enc["attention_mask"][0].tolist(),
        })
    return Dataset.from_list(rows)


class _CaptureInputWrapper(torch.nn.Module):
    """包装 backbone：在 forward 时保存 input_ids，供后续 score 解码用。"""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self._model = backbone
        self._last_input_ids = None

    def forward(self, input_ids=None, **kwargs):
        if input_ids is not None:
            self._last_input_ids = input_ids.detach().clone()
        kwargs.setdefault("output_hidden_states", True)
        return self._model(input_ids=input_ids, **kwargs)


class _SequenceRewardModelWrapper(torch.nn.Module):
    """
    适配 trl.experimental.ppo 的 reward_model：需具备 base_model_prefix 与 score(hidden_states)。
    内部用 ref_model 做 backbone（仅跑 forward 取 hidden_states），score 时用 last_input_ids
    解码后交给 SequenceRewardModel 打分。
    """

    base_model_prefix = "pretrained_model"

    def __init__(self, ref_backbone: torch.nn.Module, tokenizer: Any):
        super().__init__()
        self.pretrained_model = _CaptureInputWrapper(ref_backbone)
        self.tokenizer = tokenizer
        self._reward_model = SequenceRewardModel()

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        device = hidden_states.device
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        if getattr(self.pretrained_model, "_last_input_ids", None) is None:
            return torch.zeros(batch_size, seq_len, 1, device=device, dtype=hidden_states.dtype)
        input_ids = self.pretrained_model._last_input_ids
        rewards = []
        for i in range(input_ids.size(0)):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            dialogue = _parse_dialogue_from_text(text)
            r = self._reward_model.score(dialogue)
            rewards.append(float(r))
        r_tensor = torch.tensor(rewards, device=device, dtype=hidden_states.dtype)
        return r_tensor.view(-1, 1).expand(-1, seq_len).unsqueeze(-1)


def run_one_training(
    courses: List[dict],
    model_name: str = "Qwen/Qwen3-14B",
    max_turns: int = 50,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    output_dir: Union[str, Path] = ".",
    save_model: bool = True,
) -> dict:
    """
    使用 trl.experimental.ppo 新 API 完成一次训练：按课程列表构建 Dataset，
    用 PPOTrainer.train() 做 PPO，reward 由 SequenceRewardModel 通过包装器提供。
    """
    from datasets import Dataset as HfDataset
    from trl.experimental.ppo import (
        PPOConfig,
        PPOTrainer,
        AutoModelForCausalLMWithValueHead,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir.resolve())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 策略模型、参考模型、价值模型、奖励包装（复用 ref 做 backbone）
    policy_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    value_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    reward_model = _SequenceRewardModelWrapper(ref_model, tokenizer)
    reward_model = reward_model.to("cuda")

    # 数据集：每门课一条 prompt（第一句发言）
    train_dataset = _build_ppo_dataset(courses, tokenizer, CLASSROOM_STYLE_GUIDE)
    n_courses = len(train_dataset)
    eval_dataset = HfDataset.from_dict(train_dataset[: min(2, n_courses)]) if n_courses else train_dataset

    # 新 API：第一个参数是 args（PPOConfig），且需提供 reward_model / train_dataset / value_model
    ppo_config = PPOConfig(
        output_dir=output_dir_str,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1.41e-5,
        num_train_epochs=1,
        response_length=min(256, max_tokens),
        temperature=temperature,
        num_mini_batches=1,
        bf16=True,
        remove_unused_columns=False,
    )
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        value_model=value_model,
        eval_dataset=eval_dataset,
    )

    ppo_trainer.train()

    out_ts = time.strftime("%Y%m%d%H%M%S")
    result_path = output_dir / f"result_{out_ts}.json"
    all_results = [
        {
            "course_id": courses[i].get("id", i + 1),
            "topic": courses[i].get("topic", ""),
            "subject": courses[i].get("subject", ""),
            "grade": courses[i].get("grade", ""),
        }
        for i in range(len(courses))
    ]
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {result_path}")

    if save_model:
        model_dir = output_dir / f"ppo_{out_ts}"
        model_dir.mkdir(parents=True, exist_ok=True)
        ppo_trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        print(f"模型已保存: {model_dir}")

    return {"results": all_results, "result_path": str(result_path)}


if __name__ == "__main__":
    # 硬编码配置
    COURSE_JSON = Path(__file__).resolve().parent / "course.json"
    COURSE_IDS = None  # 使用全部课程；若只训练部分课程可写 [1, 2, 3]
    MODEL_NAME = "Qwen/Qwen3-8B"
    MAX_TURNS = 50
    MAX_TOKENS = 4096
    TEMPERATURE = 0.2
    OUTPUT_DIR = Path(__file__).resolve().parent / "result"
    SAVE_MODEL = True

    courses = load_courses(COURSE_JSON, course_ids=COURSE_IDS)
    if not courses:
        print("未找到任何课程，请检查课程文件路径或 COURSE_IDS")
        exit(1)
    print(f"共加载 {len(courses)} 门课程，开始一次训练...")

    run_one_training(
        courses=courses,
        model_name=MODEL_NAME,
        max_turns=MAX_TURNS,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        output_dir=OUTPUT_DIR,
        save_model=SAVE_MODEL,
    )
