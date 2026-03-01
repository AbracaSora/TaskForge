"""
奖励模型与动作标注：
- RewardModelBase：输入「仅包含模型回复」的 List，输出一个标量分数。
- judge_replies：输入一整段课堂对话，对每一句发言做 CI-PCD 动作标记（参考 Judger.py）。
"""

import json
import os
import re
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI
import torch

from AutomatonStruct import alphabet
from ExpertSeq import get_expert_seqs
from IRL import SequenceRewardNet, load_npa_model, propagate_npa

load_dotenv(Path(__file__).resolve().parent / ".env")


class RewardModelBase(nn.Module):
    """基类：输入模型回复列表（仅回复本身），输出整段交互的分数。"""

    @abstractmethod
    def score(self, replies: List[str]) -> float:
        """
        Args:
            replies: 一段交互中所有模型回复的列表，每个元素仅为该轮回复文本，
                     例如「老师：……」「学生：……」，不含额外前缀。
        Returns:
            标量分数。
        """
        pass


# ------------------------- CI-PCD 动作标注 -------------------------

ACTION_SET = set(alphabet)
_ACTION_UPPER = {a.upper() for a in alphabet}

ACTION_MEANINGS = """
Ipk=基础知识类提问  Pk=基础知识类回应  Ipi=个人观点类提问  Pi=个人观点类回应
Ia=分析阐释式提问  An=分析阐释式回应  Ic=总结归纳式提问  Co=总结归纳式回应
Is=迁移创新式提问  Sp=迁移创新式回应  Iu=回应建构式提问  Up=回应建构式回应
Ag=认同  Qu=质疑  Gu=指导
"""


def _create_client(
    api_key: Optional[str] = None, base_url: Optional[str] = None
) -> OpenAI:
    """从 .env 读取 OPENAI_API_KEY、OPENAI_BASE_URL 创建兼容客户端。"""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _build_judge_prompt(utterance: str, context: Optional[List[str]] = None) -> str:
    """构造单句动作判定的提示词（与 Judger.py 保持一致风格）。"""
    ctx = ""
    if context:
        ctx = "\n上文对话（供参考）：\n" + "\n".join(context[-8:]) + "\n\n"
    return f"""判断下面这条发言包含哪些 CI-PCD 教学动作，按在发言中**出现的先后顺序**列出动作代号。

动作代号及含义：
{ACTION_MEANINGS}

{ctx}当前发言：
{utterance}

要求：
- 按发言中出现的先后顺序输出，先出现的动作先写。
- 只输出一个列表，格式如 [Ag, Ipk] 或 [Gu]，不要解释、不要换行。
- 若一句中先有认同/肯定再有提问，则先写 Ag 再写 Ipk/Ia/Ic 等。"""


def _parse_action_list(text: str) -> List[str]:
    """从 LLM 输出中解析出合法动作列表并保持顺序。"""
    text = (text or "").strip()
    order: List[str] = []
    seen: set[str] = set()

    def canonical(code: str) -> Optional[str]:
        c = code.strip().strip(".")
        if not c:
            return None
        up = c.upper()
        if up in _ACTION_UPPER:
            # 按 alphabet 中的原始大小写返回
            for a in alphabet:
                if a.upper() == up:
                    return a
        return None

    # 先找方括号内的，例如 [Ag, Ipk]
    m = re.search(r"\[([^\]]*)\]", text)
    if m:
        part = m.group(1)
        for token in re.split(r"[,，\s]+", part):
            code = canonical(token)
            if code and code not in seen:
                order.append(code)
                seen.add(code)
        if order:
            return order

    # 否则按顺序扫描所有出现的合法动作（长代号优先）
    candidates = sorted(ACTION_SET, key=lambda x: -len(x))
    hits: List[tuple[int, str]] = []
    for a in candidates:
        idx = text.upper().find(a.upper())
        if idx >= 0 and a not in seen:
            hits.append((idx, a))
            seen.add(a)
    hits.sort(key=lambda x: x[0])
    return [a for _, a in hits]


def judge_replies(
    dialogue: List[str],
    *,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> List[List[str]]:
    """
    输入一整段课堂对话，对每一句发言作 CI-PCD 动作标记。

    Args:
        dialogue: 课堂对话列表，每个元素是一句完整发言，如 "老师：……" 或 "学生：……"。
        client: 可选，已创建的 OpenAI 兼容客户端；不传则自动根据环境变量创建。
        model: 判定使用的模型名称。

    Returns:
        与 dialogue 等长的列表，每个元素是当前句子的动作代号列表，例如：
        [
            ["Ag", "Ipk"],
            ["Pk"],
            ["Gu"],
            ...
        ]
    """
    if not dialogue:
        return []

    client = client or _create_client()
    history: List[str] = []
    all_actions: List[List[str]] = []

    for utterance in dialogue:
        prompt = _build_judge_prompt(utterance, context=history)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        actions = _parse_action_list(text)
        # 若解析失败，默认标为 Gu（指导），避免空标签
        if not actions:
            actions = ["Gu"]
        all_actions.append(actions)
        history.append(utterance)

    return all_actions


class SequenceRewardModel(RewardModelBase):
    """
    序列级奖励模型：输入一段交互的回复列表，输出一个标量分数。
    """

    def __init__(
        self, client: Optional[OpenAI] = None, model: str = "gemini-2.5-flash",device="cpu"
    ) -> None:
        super().__init__()
        self._client = client
        self._model_name = model
        self._npa = None
        self._reward_net = None
        self._device = device
        self.load_model(device=device)

    def forward(self, replies: List[str]) -> float:
        return self.score(replies)

    def get_actions_sequence(self, replies: List[str]) -> List[str]:
        actions_per_utt = judge_replies(
            replies, client=self._client, model=self._model_name
        )
        return actions_per_utt

    def score(self, replies: List[str]) -> float:
        actions_per_utt = judge_replies(
            replies, client=self._client, model=self._model_name
        )
        print(actions_per_utt)
        actions_index = [
            self._alphabet_to_index[a] for action in actions_per_utt for a in action
        ]
        actions_tensor = torch.tensor(
            [actions_index],
            dtype=torch.long,
            device=self._device,
        )
        state_dists = torch.tensor(
            [[1.0] + [0.0] * (self._npa.S - 1)],
            device=self._device,
        )
        state_dists = propagate_npa(self._npa, actions_tensor, state_dists)
        score = self._reward_net(state_dists, actions_tensor)
        score /= 100
        return score.item()

    def actions_to_score(self, actions: List[str]) -> float:
        actions_index = [self._alphabet_to_index[a] for a in actions]
        actions_tensor = torch.tensor(
            [actions_index],
            dtype=torch.long,
            device=self._device,
        )
        state_dists = torch.tensor(
            [[1.0] + [0.0] * (self._npa.S - 1)],
            device=self._device,
        )
        state_dists = propagate_npa(self._npa, actions_tensor, state_dists)
        score = self._reward_net(state_dists, actions_tensor)
        return score.item()

    def load_model(
        self,
        npa_path: str = "model/npa.pt",
        reward_net_path: str = "model/reward_net.pt",
        device: Optional[str] = None,
    ) -> None:

        if self._npa is None:
            self._npa, _ = load_npa_model(npa_path)
        self._npa.to(device or self._device)
        self._npa.eval()

        if self._reward_net is None:
            num_states = self._npa.S
            num_actions = len(alphabet)
            self._reward_net = SequenceRewardNet(num_states, num_actions)
            state_dict = torch.load(reward_net_path, map_location=device or self._device)
            self._reward_net.load_state_dict(state_dict)
        self._reward_net.to(device or self._device)
        self._reward_net.eval()

        self._alphabet_to_index = {a: i for i, a in enumerate(sorted(alphabet))}

    def get_reward_with_reward_net(
        self,
        action_sequence: Union[List[str], List[List[str]]],
    ) -> float:
        """
        使用 RewardNet 对给定动作序列计算序列级奖励值。
        动作序列会先经 NPA 做状态分布传播，再输入序列级奖励网络得到标量 R(τ)。

        :param action_sequence: 动作代号序列。可为扁平列表如 ["Ag", "Ipk", "Gu", ...]，
            或按句划分的列表如 [["Ag", "Ipk"], ["Gu"], ...]，内部会按顺序展平。
        :param npa: 已加载的 NPA 模型；为 None 时从 npa_path 加载。
        :param reward_net: 已加载的 SequenceRewardNet；为 None 时从 reward_net_path 加载。
        :param npa_path: NPA 权重路径。
        :param reward_net_path: 奖励网络权重路径。
        :param device: 运行设备，如 "cpu" / "cuda"；None 时与 reward_net 一致。
        :return: 该轨迹的奖励标量；轨迹长度 < 2 时返回 None。
        """
        # 展平：支持 [["Ag","Ipk"], ["Gu"]] -> ["Ag","Ipk","Gu"]
        flat = []
        for x in action_sequence:
            if isinstance(x, (list, tuple)):
                flat.extend(x)
            else:
                flat.append(x)
        action_sequence = flat

        alphabet_sorted = sorted(alphabet)
        alphabet_to_index = {a: i for i, a in enumerate(alphabet_sorted)}
        indices = [alphabet_to_index.get(a, 0) for a in action_sequence]
        if len(indices) < 2:
            return None

        dev = self._device

        traj = torch.tensor([indices], dtype=torch.long, device=dev)
        init_dist = torch.tensor([[1.0] + [0.0] * (self._npa.S - 1)], device=dev)
        state_dists = propagate_npa(self._npa, traj, init_dist)
        with torch.no_grad():
            r = self._reward_net(state_dists, traj)
        return r.squeeze().item()


if __name__ == "__main__":
    from task import ClassroomSceneGenerationTask
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # task = ClassroomSceneGenerationTask()
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", device_map="auto")
    # messages = task.run(model=model, tokenizer=tokenizer)
    messages = json.load(
        open("result/result_20260227045600.json", "r", encoding="utf-8")
    )
    replies = messages["dialogue"]
    reward_model = SequenceRewardModel()
    print(reward_model.score(replies))
