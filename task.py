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
        temperature: float = 0.7,
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

        model_inputs = tok([text], return_tensors="pt").to(
            model.device
        )
        prompt_tokens = model_inputs["input_ids"].shape[-1]
        with torch.no_grad():
            generated_ids = m.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.95,
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
# 专业课程结构参考：4703_数学_初三.json（相似三角形的周长比与面积比）等真实课堂 transcript
CLASSROOM_STYLE_GUIDE = """
你是一名教育专家，正在模拟一堂真实课堂的对话，根据已有对话生成**唯一的下一条**发言。参考真实课堂的课程结构。

【课程结构（参考专业课堂实录）】
- 开场：学生「老师好。」→ 老师「同学们好，请坐。我们前面学了…请大家回顾一下…你来说。」
- 回顾与引入：老师可连续多句（如先请学生回顾旧知，再引入今日课题、读题或讲解题干）。
- 师生问答：老师提问→学生答→老师可**连续发言**（如先反馈「非常好」「是的」再追问「理由呢？」「等于多少？」；或先讲解/读题再追问「相似吗？为什么？」）；再学生答→老师总结或提新问。
- 同一角色可连续发言：老师可连续多条（讲解、追问、点名、认同、总结）；学生也可连续（如多人依次回答时）。由上下文自然决定谁接着说话。
- 推进与结束：课堂应有递进（回顾→新知识→例题/练习→小结），结尾老师布置作业或「下课」，学生「老师再见」。

【发言类型参考】
- 老师：回顾提问、讲解/读题、追问（「理由呢？」「对吗？」「等于多少？」）、认同（「非常好」「是的」）、指导（「好，张毅。」）、总结（「我们把这个过程整理整理…所以？」）。可点名、可连续讲解再追问。
- 学生：简洁作答（「对。」「1:2。」「相似比。」）或稍长解释（推理、读题结论）。可表示不懂（「能再讲一下吗？」）。

【格式与禁止】
- 仅输出一行，格式：老师：<内容> 或 学生：<内容>。根据上下文决定发言者，同一角色可连续。
- 禁止占位符、禁止「根据对话我生成…」等元语言。不要无意义重复同一问句或同一句话。
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
        temperature: float = 0.8,
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

        # initial_prompt = self._parse_messages_to_prompt(
        #     self.full_dialogue[:1], tokenizer
        # )["text"]
        # query_t = tokenizer(initial_prompt, return_tensors="pt").to("cuda")
        # response = self._parse_messages_to_prompt(self.full_dialogue[1:], tokenizer)[
        #     "text"
        # ]
        # response_t = tokenizer(response, return_tensors="pt").to("cuda")
        # reward = self.reward_model.score(dialogue)

        return {
            "dialogue": dialogue,
            "full_dialogue": self.full_dialogue,
            "topic": self.topic,
            "subject": self.subject,
            "grade": self.grade,
            # "query_tensors": query_t,
            # "response_tensors": response_t,
            # "reward": reward,
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
        ctx = "\n".join(dialogue_so_far[-14:]) if dialogue_so_far else ""
        course_line = (
            f"\n当前课堂：{' '.join(p for p in [subject, grade, topic] if p)}\n"
            if (subject or grade or topic)
            else ""
        )
        prompt = f"""根据下面已有对话，直接输出**下一条**发言（仅一条），不要有任何分析、说明或复述题目。
{course_line}
已有对话：
{ctx}

输出要求：
- 只输出一行，格式：老师：<内容> 或 学生：<内容>。**由上下文自然决定**是老师还是学生；**同一角色可以连续发言**（例如老师连续讲解与追问、先反馈再提新问，或学生多人依次回答）。
- 老师可：回顾提问、读题/讲解、追问（理由呢？等于多少？）、认同（非常好）、点名、总结。学生可极简短（「对。」「1:2。」）或稍长解释。内容贴合当前科目与主题，课堂有递进感。
- 禁止占位符与元语言。不要无意义重复同一问句。直接以「老师：」或「学生：」开头。"""
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


if __name__ == "__main__":
    import json
    DATA_DIR = "./expert_sample"
    OUTPUT_DIR = "./non_expert_sample_out"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = os.listdir(DATA_DIR)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
    for file in files:
        with open(os.path.join(DATA_DIR, file), "r") as f:
            data = json.load(f)
            task = ClassroomSceneGenerationTask(topic=data["name"], subject=data["subject"], grade=data["grade"])
            out = task.run(model, tokenizer)
            with open(os.path.join(OUTPUT_DIR, file.replace(".json", "_out.json")), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=4)