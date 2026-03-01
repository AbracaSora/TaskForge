"""
Judger：判断当前发言包含的 CI-PCD 动作，按在发言中出现的先后顺序输出列表。
若一句含多个动作，则返回 [动作1, 动作2, ...]（按出现顺序）。
配置：.env 中设置 OPENAI_API_KEY、OPENAI_BASE_URL
另：提供使用 RewardNet 对动作序列计算序列级奖励的接口。
"""

import json
import os
import re
from typing import Optional, Union

import torch
from dotenv import load_dotenv
from openai import OpenAI

from Agent import run_session
from AutomatonStruct import alphabet
from ExpertSeq import get_expert_seqs
from IRL import SequenceRewardNet, load_npa_model, propagate_npa

load_dotenv()

ACTION_SET = set(alphabet)
# 大写到规范代号的映射，便于解析时统一返回格式
_ACTION_UPPER = {a.upper(): a for a in alphabet}


def _create_client(
    api_key: Optional[str] = None, base_url: Optional[str] = None
) -> OpenAI:
    """从 .env 读取 OPENAI_API_KEY、OPENAI_BASE_URL。"""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


# CI-PCD 动作含义简表（供 LLM 判断）
ACTION_MEANINGS = """
Ipk=基础知识类提问  Pk=基础知识类回应  Ipi=个人观点类提问  Pi=个人观点类回应
Ia=分析阐释式提问  An=分析阐释式回应  Ic=总结归纳式提问  Co=总结归纳式回应
Is=迁移创新式提问  Sp=迁移创新式回应  Iu=回应建构式提问  Up=回应建构式回应
Ag=认同  Qu=质疑  Gu=指导
"""


def _build_judge_prompt(utterance: str, context: Optional[list[str]] = None) -> str:
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


def _parse_action_list(text: str) -> list[str]:
    """从模型输出中解析出合法动作列表并保持顺序。"""
    text = text.strip()
    # 匹配 [...] 或 用逗号/空格分隔的代号
    order: list[str] = []
    seen: set[str] = set()

    def canonical(code: str) -> Optional[str]:
        return _ACTION_UPPER.get(code.upper()) if code else None

    # 先找方括号内的
    m = re.search(r"\[([^\]]*)\]", text)
    if m:
        part = m.group(1)
        for token in re.split(r"[,，\s]+", part):
            code = canonical(token.strip().strip("."))
            if code and code not in seen:
                order.append(code)
                seen.add(code)
        if order:
            return order
    # 否则按顺序扫描所有出现的合法动作
    for a in sorted(ACTION_SET, key=lambda x: -len(x)):  # 长代号优先
        idx = text.upper().find(a.upper())
        if idx >= 0 and a not in seen:
            order.append((idx, a))
            seen.add(a)
    return [a for _, a in sorted(order, key=lambda x: x[0])]


def judge_utterance(
    utterance: str,
    context: Optional[list[str]] = None,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o",
) -> list[str]:
    """
    判断当前发言包含的动作，按在发言中出现的先后顺序返回列表。
    例如：「老师：很好，你准确地说出了三种解法。现在，谁能详细解释一下因式分解法？」 -> [Ag, Ipk]

    :param utterance: 当前发言，如 "老师：……"
    :param context: 可选，上文对话列表，用于消歧
    :param client: 可选，OpenAI 兼容客户端
    :param model: 模型名
    :return: 动作代号列表，如 ["Ag", "Ipk"]
    """
    client = client or _create_client()
    prompt = _build_judge_prompt(utterance, context=context)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    result = _parse_action_list(text)
    return result if result else ["Gu"]


def get_reward_with_reward_net(
    action_sequence: Union[list[str], list[list[str]]],
    npa=None,
    reward_net=None,
    npa_path: str = "model/npa.pt",
    reward_net_path: str = "model/reward_net.pt",
    device: Optional[str] = None,
) -> Optional[float]:
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

    if npa is None:
        npa, _ = load_npa_model(npa_path)
    npa.eval()

    if reward_net is None:
        num_states = npa.S
        num_actions = len(alphabet)
        reward_net = SequenceRewardNet(num_states, num_actions)
        state_dict = torch.load(reward_net_path, map_location="cpu")
        reward_net.load_state_dict(state_dict)
    reward_net.eval()

    dev = device or next(reward_net.parameters()).device
    npa = npa.to(dev)
    reward_net = reward_net.to(dev)
    traj = torch.tensor([indices], dtype=torch.long, device=dev)
    init_dist = torch.tensor([[1.0] + [0.0] * (npa.S - 1)], device=dev)
    state_dists = propagate_npa(npa, traj, init_dist)
    with torch.no_grad():
        r = reward_net(state_dists, traj)
    return r.squeeze().item()


def generate_non_expert_trajectory_by_agent(
    topic: str = "一元二次方程的解法",
    subject: str = "数学",
    grade: str = "初中二年级",
    min_len: int = 100,
    max_len: int = 150,
    max_turns_per_session: int = 120,
    client_agent=None,
    client_judger=None,
    model_agent: str = "gpt-4o-mini",
    model_judger: str = "gpt-4o",
) -> Optional[list[str]]:
    """
    用 Agent 生成课堂对话、Judger 对每句标注动作，得到长度在 [min_len, max_len] 的动作序列（CI-PCD 代号列表）。
    若单轮对话动作数不足 min_len，会继续跑新轮并拼接，直到不少于 min_len 再截断到 max_len。
    """
    client = client_judger or client_agent or _create_client()
    actions: list[str] = []

    while len(actions) < min_len:
        dialogue = run_session(
            topic,
            subject,
            grade,
            max_turns=max_turns_per_session,
            client=client_agent or client,
            model=model_agent,
        )
        history: list[str] = []
        for utterance in dialogue:
            a_list = judge_utterance(
                utterance,
                context=history,
                client=client_judger or client,
                model=model_judger,
            )
            history.append(utterance)
            actions.extend(a_list)
        if len(actions) >= max_len:
            break

    if len(actions) < 2:
        return None
    return actions, dialogue


# ---------------------------------------------------------------------------
# 示例
# ---------------------------------------------------------------------------
def main():
    expert_sample = get_expert_seqs()[0]
    print(expert_sample)
    reward = get_reward_with_reward_net(expert_sample)
    print(reward)
    non_expert_sample = json.load(open("non_expert/001.json", "r", encoding="utf-8"))[
        "trajectory"
    ]
    print(non_expert_sample)
    reward = get_reward_with_reward_net(non_expert_sample)
    print(reward)


if __name__ == "__main__":
    main()
