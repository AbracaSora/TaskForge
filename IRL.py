"""
IRL：NPA 作为奖励结构，学习序列级奖励函数

流程：
  1. NPA 提供状态空间、转移结构、状态表示（state_dist）
  2. 序列级奖励 R_theta(trajectory) -> 标量
  3. 从专家 demonstration 与 non_expert 对比学习，使专家轨迹获得更高回报
"""

import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from AutomatonStruct import alphabet, states, transitions, start_state
from NeuralProbabilisticAutomaton import NPA


# -----------------------------------------------------------------------------
# 1. 数据加载
# -----------------------------------------------------------------------------


def load_action_list(data_dir: str = "./action_list_single") -> list:
    """
    从目录加载专家动作序列。
    每文件为 JSON，需含 "actions" 键（动作名列表）。
    """
    seqs = []
    if not os.path.isdir(data_dir):
        return seqs
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            seqs.append(data["actions"])
        except Exception:
            continue
    return seqs


def load_non_expert_from_dir(
    non_expert_dir: str = "non_expert",
    alphabet_to_index: dict = None,
    max_len: int = 150,
    device: str = "cpu",
) -> list:
    """
    从目录加载非专家轨迹（Agent 生成并 Judger 标注的 JSON）。
    每文件需含 "trajectory" 或 "actions"（动作名列表）。
    返回 list of tensor [1, T]，长度 < 2 的轨迹会跳过。
    """
    if alphabet_to_index is None:
        alphabet_to_index = {a: i for i, a in enumerate(sorted(alphabet))}
    tensors = []
    if not os.path.isdir(non_expert_dir):
        return tensors
    for fname in sorted(os.listdir(non_expert_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(non_expert_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            traj = data.get("trajectory", data.get("actions", []))
        except Exception:
            continue
        indices = [alphabet_to_index.get(a, 0) for a in traj[:max_len]]
        if len(indices) < 2:
            continue
        tensors.append(torch.tensor([indices], dtype=torch.long, device=device))
    return tensors


def load_npa_model(checkpoint_path: str = "model/npa.pt"):
    """加载训练好的 NPA 模型，返回 (model, alphabet_to_index)。"""
    alphabet_to_index = {a: i for i, a in enumerate(sorted(alphabet))}
    state_to_index = {s: i for i, s in enumerate(sorted(states))}
    accept_states = {state_to_index[s] for s in ["q8"]}
    delta = {
        state_to_index[q]: {
            alphabet_to_index[a]: {state_to_index[q2] for q2 in next_states}
            for a, next_states in a_dict.items()
        }
        for q, a_dict in transitions.items()
    }
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = NPA(
        num_states=ckpt["num_states"],
        num_actions=ckpt["num_actions"],
        delta_dict=delta,
        accept_states=accept_states,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, alphabet_to_index


# -----------------------------------------------------------------------------
# 2. NPA 传播与长度奖励工具
# -----------------------------------------------------------------------------


def propagate_npa(npa, traj_tensor, init_dist, eps: float = 1e-8) -> torch.Tensor:
    """
    用 NPA 沿轨迹传播状态分布。
    traj_tensor: [B, T] long
    init_dist: [B, S]
    返回 state_dists [B, T+1, S]，含初始分布。
    """
    npa.eval()
    B, T = traj_tensor.shape
    S = init_dist.shape[1]
    device = traj_tensor.device
    state_dist = init_dist.to(device)
    dists = [state_dist]

    with torch.no_grad():
        for t in range(T):
            a = traj_tensor[:, t]
            all_prob = []
            for q in range(S):
                s_vec = torch.full((B,), q, device=device, dtype=torch.long)
                prob = npa.transition_prob(s_vec, a)
                all_prob.append(prob.unsqueeze(1))
            all_prob = torch.cat(all_prob, dim=1)
            next_dist = torch.einsum("bq,bqs->bs", state_dist, all_prob)
            state_dist = next_dist / (next_dist.sum(dim=-1, keepdim=True) + eps)
            dists.append(state_dist)

    return torch.stack(dists, dim=0).permute(1, 0, 2)


def length_similarity_bonus(T: float, T_ref: float, sigma: float, coef: float) -> float:
    """
    长度相似性加分：T 越接近 T_ref 越大，偏离越远越小（高斯）。
    评估时若希望「接近专家长度的轨迹奖励更高」，可在 R_net 输出上加上本项。
    """
    return coef * math.exp(-((T - T_ref) ** 2) / (2 * sigma**2))


# -----------------------------------------------------------------------------
# 3. 序列级奖励网络
# -----------------------------------------------------------------------------


class SequenceRewardNet(nn.Module):
    """
    序列级奖励 R_theta(trajectory) -> 标量。
    输入：state_dists [B, T+1, S]，actions [B, T]
    输出：[B] 每条轨迹一个标量，取值 [0, reward_scale]
    """

    def __init__(self, num_states, num_actions, hidden_dim=64, reward_scale=100.0):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward_scale = reward_scale
        self.action_embed = nn.Embedding(num_actions, 16)
        self.lstm = nn.LSTM(
            num_states + 16, hidden_dim, batch_first=True, bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_dists, actions):
        """state_dists [B, T+1, S]，actions [B, T] -> [B]"""
        B, T_plus, S = state_dists.shape
        T = T_plus - 1
        a_emb = self.action_embed(actions)
        s_t = state_dists[:, :T]
        x = torch.cat([s_t, a_emb], dim=-1)
        out, (h_n, _) = self.lstm(x)
        logit = self.fc(h_n[-1]).squeeze(-1)
        return torch.sigmoid(logit) * self.reward_scale


# -----------------------------------------------------------------------------
# 4. IRL 奖励学习
# -----------------------------------------------------------------------------

# 训练超参：loss = relu(MARGIN - (R_expert_adj - R_non_adj))，当差值≥MARGIN 时 loss=0
# 若 loss 很快降为 0，说明奖励网已能区分专家/非专家；可增大 MARGIN 或降低 lr 使训练更平滑
MARGIN = 15.0


def _adjusted_reward(
    R_net,
    T_len,
    length_bonus_coef: float,
    length_similarity_bonus_fn,
) -> torch.Tensor:
    """R_adj = R_net + length_bonus(log(1+T)) + length_similarity_bonus(T)"""
    return (
        R_net
        + length_bonus_coef * math.log(1 + T_len)
        + length_similarity_bonus_fn(T_len)
    )


def irl_learn_reward(
    npa,
    expert_seqs,
    alphabet_to_index,
    max_epochs=10,
    lr=0.0001,
    max_len=200,
    non_expert_dir: str = "non_expert",
    length_bonus_coef=1.0,
    length_similarity_coef=5.0,
    length_similarity_sigma=None,
):
    """
    从专家 vs 非专家轨迹学习序列级奖励 R_theta。

    非专家轨迹仅从 non_expert_dir 下 JSON（含 "trajectory"）加载。
    奖励调整：
      - length_bonus_coef: R += coef * log(1+T)（长序列补偿）
      - length_similarity_coef / sigma: R += coef * exp(-(T-T_ref)^2/(2*sigma^2))（接近专家长度加分）
    """
    device = next(npa.parameters()).device
    num_states = npa.S
    init_dist = torch.tensor([[1.0] + [0.0] * (num_states - 1)], device=device)

    # 专家 -> tensor 列表
    expert_tensors = []
    for seq in expert_seqs:
        indices = [alphabet_to_index.get(a, 0) for a in seq[:max_len]]
        if len(indices) < 2:
            continue
        expert_tensors.append(torch.tensor([indices], dtype=torch.long, device=device))

    # 专家长度参考（用于长度相似性加分）
    expert_lengths = [t.shape[1] for t in expert_tensors]
    T_ref = sum(expert_lengths) / len(expert_lengths) if expert_lengths else 50.0
    sigma = length_similarity_sigma
    if sigma is None:
        sigma = max(20.0, 0.25 * T_ref)

    def _length_similarity(T: float) -> float:
        return length_similarity_coef * math.exp(-((T - T_ref) ** 2) / (2 * sigma**2))

    # 非专家轨迹（从目录加载）
    non_expert_tensors = load_non_expert_from_dir(
        non_expert_dir, alphabet_to_index, max_len, device
    )
    if not non_expert_tensors:
        raise FileNotFoundError(
            f"未在 {non_expert_dir}/ 下找到有效非专家轨迹，请先运行 Agent.py batch 生成"
        )

    def _sample_non_expert(n: int):
        return [random.choice(non_expert_tensors) for _ in range(n)]

    reward_net = SequenceRewardNet(num_states, len(alphabet)).to(device)
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr, weight_decay=1e-4)

    print(
        f"IRL: {len(expert_tensors)} 条专家, {len(non_expert_tensors)} 条非专家 (来自 {non_expert_dir}/)"
    )
    print(
        f"  T_ref={T_ref:.0f}, sigma={sigma:.0f}, length_similarity_coef={length_similarity_coef}"
    )

    for epoch in range(max_epochs):
        total_loss = 0.0
        n_pairs = 0
        sum_ret_expert = 0.0
        sum_ret_non = 0.0

        for ex_traj in expert_tensors:
            non_list = _sample_non_expert(1)
            if not non_list:
                continue
            non_traj = non_list[0]

            if ex_traj.shape[1] < 2 or non_traj.shape[1] < 2:
                continue
            # 不截断：专家与非专家各自以全长进入 NPA 与奖励网络
            ex_t = ex_traj
            non_t = non_traj

            state_dists_expert = propagate_npa(npa, ex_t, init_dist)
            state_dists_non = propagate_npa(npa, non_t, init_dist)

            ret_expert = reward_net(state_dists_expert, ex_t)
            ret_non = reward_net(state_dists_non, non_t)

            T_expert = ex_traj.shape[1]
            T_non = non_traj.shape[1]
            ret_expert_adj = _adjusted_reward(
                ret_expert, T_expert, length_bonus_coef, _length_similarity
            )
            ret_non_adj = _adjusted_reward(
                ret_non, T_non, length_bonus_coef, _length_similarity
            )

            loss = F.relu(MARGIN - (ret_expert_adj - ret_non_adj)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_pairs += 1
            with torch.no_grad():
                sum_ret_expert += ret_expert.mean().item()
                sum_ret_non += ret_non.mean().item()

        if n_pairs > 0 and epoch % 10 == 0:
            avg_ret_expert = sum_ret_expert / n_pairs
            avg_ret_non = sum_ret_non / n_pairs
            print(
                f"  Epoch {epoch} IRL loss: {total_loss / n_pairs:.4f}  "
                f"(R_expert≈{avg_ret_expert:.1f}, R_non≈{avg_ret_non:.1f}, diff≈{avg_ret_expert - avg_ret_non:.1f})"
            )

    return reward_net


# -----------------------------------------------------------------------------
# 5. 主流程
# -----------------------------------------------------------------------------


def main():
    print("=" * 50)
    print("IRL：NPA 作为奖励结构，学习奖励函数")
    print("=" * 50)

    if not os.path.exists("model/npa.pt"):
        print("请先运行 NeuralProbabilisticAutomaton.py 训练并保存 NPA 模型")
        return

    npa, alphabet_to_index = load_npa_model()
    npa.to("cuda")
    npa.eval()

    expert_seqs = load_action_list()
    print(f"加载 {len(expert_seqs)} 条专家序列")

    reward_net = irl_learn_reward(
        npa,
        expert_seqs,
        alphabet_to_index,
        max_epochs=10,
        non_expert_dir="non_expert",
        length_similarity_coef=0,
    )

    os.makedirs("model", exist_ok=True)
    torch.save(reward_net.state_dict(), "model/reward_net.pt")
    print("奖励网络已保存至 model/reward_net.pt")


if __name__ == "__main__":
    main()
