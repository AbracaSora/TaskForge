"""
IRL 示例：NPA 作为奖励结构，学习奖励函数

流程：
1. NPA 提供状态空间、转移结构、状态表示（state_dist）
2. 序列级奖励 R_theta(trajectory) -> 标量
3. 从专家 demonstration 中学习 theta，使专家轨迹获得更高回报
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from AutomatonStruct import alphabet, states, transitions, start_state
from NeuralProbabilisticAutomaton import NPA

# 动作名列表，用于随机采样
ALPHABET_LIST = sorted(alphabet)


# =====================================================
# 非专家轨迹生成（多种策略，更适合 IRL 对比学习）
# =====================================================
def nfa_random_walk(alphabet_to_index, max_len=150, accept_state="q8", min_len=2):
    """
    NFA 上随机游走：从 start_state 沿合法转移随机走，直到到达 accept 或达 max_len。
    得到「结构合法但顺序随机」的轨迹，与专家路径形成对比。
    """
    state = start_state
    path = []
    for _ in range(max_len):
        if state == accept_state:
            break
        if state not in transitions:
            break
        valid_actions = list(transitions[state].keys())
        if not valid_actions:
            break
        action = random.choice(valid_actions)
        next_states = transitions[state][action]
        state = random.choice(next_states)
        path.append(action)
    indices = [alphabet_to_index.get(a, 0) for a in path]
    return indices if len(indices) >= min_len else None


def perturb_expert(
    seq,
    alphabet_to_index,
    strategy="swap",
    replace_ratio=0.15,
    perturb_len_ratio=0.5,
):
    """
    对专家序列做轻度扰动，得到「接近专家但更差」的负样本，利于学细粒度偏好。
    strategy: 'swap' | 'replace' | 'shuffle_suffix' | 'shuffle_prefix' | 'insert' | 'delete' | 'full_shuffle'
    """
    seq = list(seq)
    if len(seq) < 2:
        return None
    if strategy == "full_shuffle":
        out = seq.copy()
        random.shuffle(out)
    elif strategy == "swap":
        out = seq.copy()
        i, j = random.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    elif strategy == "replace":
        out = seq.copy()
        n = max(1, int(len(out) * replace_ratio))
        for _ in range(n):
            out[random.randint(0, len(out) - 1)] = random.choice(ALPHABET_LIST)
    elif strategy == "shuffle_suffix":
        out = seq.copy()
        k = max(1, int(len(out) * perturb_len_ratio))
        suffix = out[k:]
        random.shuffle(suffix)
        out = out[:k] + suffix
    elif strategy == "shuffle_prefix":
        out = seq.copy()
        k = max(1, int(len(out) * perturb_len_ratio))
        prefix = out[:k]
        random.shuffle(prefix)
        out = prefix + out[k:]
    elif strategy == "insert":
        out = seq.copy()
        pos = random.randint(0, len(out))
        out.insert(pos, random.choice(ALPHABET_LIST))
    elif strategy == "delete":
        out = seq.copy()
        if len(out) <= 2:
            return None
        out.pop(random.randint(0, len(out) - 1))
    else:
        out = seq.copy()
        random.shuffle(out)
    indices = [alphabet_to_index.get(a, 0) for a in out]
    return indices if len(indices) >= 2 else None


def segment_mix(expert_seqs, alphabet_to_index, max_len=150, min_len=2):
    """
    从多条专家轨迹各取一段拼接，得到「局部合理但整体顺序错乱」的负样本。
    """
    if len(expert_seqs) < 2:
        return None
    segs = []
    total = 0
    for _ in range(random.randint(2, 4)):
        s = random.choice(expert_seqs)
        s = s[:max_len]
        if not s:
            continue
        i = random.randint(0, max(0, len(s) - 1))
        j = random.randint(i + 1, len(s)) if i + 1 < len(s) else len(s)
        segs.extend(s[i:j])
        total += j - i
        if total >= max_len:
            break
    out = (segs + [])[:max_len]
    indices = [alphabet_to_index.get(a, 0) for a in out]
    return indices if len(indices) >= min_len else None


def sample_non_expert_indices(expert_seqs, alphabet_to_index, max_len, strategy="mixed", device="cpu"):
    """
    按策略采样一条非专家轨迹的 action indices，失败时返回 None。
    strategy: 'mixed'（推荐）| 'shuffle' | 'nfa' | 'perturb' | 'segment'
    """
    if strategy == "nfa":
        indices = nfa_random_walk(alphabet_to_index, max_len=max_len)
    elif strategy == "shuffle":
        seq = random.choice(expert_seqs)[:max_len]
        shuffled = seq.copy()
        random.shuffle(shuffled)
        indices = [alphabet_to_index.get(a, 0) for a in shuffled]
        if len(indices) < 2:
            indices = None
    elif strategy == "perturb":
        seq = random.choice(expert_seqs)[:max_len]
        sub = random.choice(["swap", "replace", "shuffle_suffix", "shuffle_prefix", "full_shuffle"])
        indices = perturb_expert(seq, alphabet_to_index, strategy=sub)
    elif strategy == "segment":
        indices = segment_mix(expert_seqs, alphabet_to_index, max_len=max_len)
    else:
        # mixed: 以一定比例从 nfa / perturb / shuffle / segment 中采样
        r = random.random()
        if r < 0.35:
            indices = nfa_random_walk(alphabet_to_index, max_len=max_len)
        elif r < 0.65:
            seq = random.choice(expert_seqs)[:max_len]
            sub = random.choice(["swap", "replace", "shuffle_suffix", "full_shuffle"])
            indices = perturb_expert(seq, alphabet_to_index, strategy=sub)
        elif r < 0.85:
            seq = random.choice(expert_seqs)[:max_len]
            shuffled = seq.copy()
            random.shuffle(shuffled)
            indices = [alphabet_to_index.get(a, 0) for a in shuffled]
            if len(indices) < 2:
                indices = nfa_random_walk(alphabet_to_index, max_len=max_len)
        else:
            indices = segment_mix(expert_seqs, alphabet_to_index, max_len=max_len)
    if indices is None or len(indices) < 2:
        return None
    return torch.tensor([indices], dtype=torch.long, device=device)


# =====================================================
# 加载
# =====================================================
def load_action_list(data_dir="./action_list_single"):
    """从 action_list_single 加载所有动作序列"""
    seqs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname), "r") as f:
            data = json.load(f)
            seqs.append(data["actions"])
    return seqs


def load_npa_model(checkpoint_path="model/npa.pt"):
    """加载训练好的 NPA 模型"""
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


def propagate_npa(npa, traj_tensor, init_dist, eps=1e-8):
    """
    用 NPA 传播轨迹，返回每步的 state_dist 序列
    traj_tensor: [B, T] long
    return: state_dists [B, T+1, S]，含初始分布
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


# =====================================================
# 序列级奖励模型：R_theta(trajectory) -> 标量
# =====================================================
class SequenceRewardNet(nn.Module):
    """
    序列级奖励 R_theta(trajectory) -> 标量
    输入：state_dists [B, T+1, S]，actions [B, T]
    输出：[B] 每条轨迹一个标量奖励
    """

    def __init__(self, num_states, num_actions, hidden_dim=64):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
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
        """
        state_dists: [B, T+1, S] 含初始分布
        actions: [B, T] long
        return: [B]
        """
        B, T_plus, S = state_dists.shape
        T = T_plus - 1

        a_emb = self.action_embed(actions)
        s_t = state_dists[:, :T]
        x = torch.cat([s_t, a_emb], dim=-1)

        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden).squeeze(-1)


def irl_learn_reward(
    npa,
    expert_seqs,
    alphabet_to_index,
    max_epochs=50,
    lr=0.001,
    max_len=150,
    non_expert_strategy="mixed",
):
    """
    从专家轨迹学习奖励函数 R_theta
    目标：专家轨迹的折扣回报 > 非专家轨迹
    non_expert_strategy: 'mixed'（推荐）| 'shuffle' | 'nfa' | 'perturb' | 'segment'
    """
    device = next(npa.parameters()).device
    num_actions = len(alphabet)
    num_states = npa.S
    init_dist = torch.tensor([[1.0] + [0.0] * (num_states - 1)], device=device)

    expert_tensors = []
    for seq in expert_seqs:
        indices = [alphabet_to_index.get(a, 0) for a in seq[:max_len]]
        if len(indices) < 2:
            continue
        expert_tensors.append(torch.tensor([indices], dtype=torch.long, device=device))

    reward_net = SequenceRewardNet(num_states, num_actions).to(device)
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr)

    def get_non_expert_trajectories(n):
        """生成非专家轨迹，策略由 non_expert_strategy 指定"""
        non_expert = []
        for _ in range(n * 3):
            if len(non_expert) >= n:
                break
            t = sample_non_expert_indices(
                expert_seqs, alphabet_to_index, max_len,
                strategy=non_expert_strategy, device=device,
            )
            if t is not None:
                non_expert.append(t)
        return non_expert[:n]

    print(f"IRL 学习奖励: {len(expert_tensors)} 条专家轨迹, 非专家策略: {non_expert_strategy}")

    for epoch in range(max_epochs):
        total_loss = 0.0
        n_pairs = 0

        for ex_traj in expert_tensors:
            non_trajs = get_non_expert_trajectories(1)
            if not non_trajs:
                continue
            non_traj = non_trajs[0]

            min_T = min(ex_traj.shape[1], non_traj.shape[1])
            if min_T < 2:
                continue
            ex_t = ex_traj[:, :min_T]
            non_t = non_traj[:, :min_T]

            state_dists_expert = propagate_npa(npa, ex_t, init_dist)
            state_dists_non = propagate_npa(npa, non_t, init_dist)

            ret_expert = reward_net(state_dists_expert, ex_t)
            ret_non = reward_net(state_dists_non, non_t)

            margin = 0.1
            loss = F.relu(margin - (ret_expert - ret_non)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_pairs += 1

        if n_pairs > 0 and epoch % 10 == 0:
            print(f"Epoch {epoch} IRL loss: {total_loss / n_pairs:.4f}")

    return reward_net


# =====================================================
# 主流程
# =====================================================
def main():
    print("=" * 50)
    print("IRL：NPA 作为奖励结构，学习奖励函数")
    print("=" * 50)

    if not os.path.exists("model/npa.pt"):
        print("请先运行 NeuralProbabilisticAutomaton.py 训练并保存模型")
        return

    npa, alphabet_to_index = load_npa_model()
    npa.eval()

    expert_seqs = load_action_list()
    print(f"加载 {len(expert_seqs)} 条专家序列")

    reward_net = irl_learn_reward(npa, expert_seqs, alphabet_to_index, max_epochs=50)

    os.makedirs("model", exist_ok=True)
    torch.save(reward_net.state_dict(), "model/reward_net.pt")
    print("奖励网络已保存至 model/reward_net.pt")


if __name__ == "__main__":
    main()
