import os
import random
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyparsing import actions

from AutomatonStruct import alphabet, states, transitions
from ExpertSeq import get_expert_seqs


class NPA(nn.Module):
    """
    Neural Probabilistic Automaton
    支持：
    - NFA结构先验
    - KL结构保持
    - 熵正则
    """

    def __init__(
        self,
        num_states,
        num_actions,
        delta_dict,
        kl_lambda=0.5,
        entropy_coef=0.05,
        illegal_bias=-5.0,
        accept_states=None,
    ):
        super().__init__()

        self.S = num_states
        self.A = num_actions
        self.kl_lambda = kl_lambda
        self.entropy_coef = entropy_coef
        self.accept_states = accept_states

        # ===== 可训练参数 =====
        self.theta = nn.Parameter(torch.zeros(self.S, self.A, self.S))

        # ===== 构造结构先验 =====
        mask, prior = self._build_structure(delta_dict, illegal_bias)
        self.register_buffer("mask", mask)
        self.register_buffer("prior", prior)

    # -------------------------------------------------
    # 构造NFA结构
    # -------------------------------------------------
    def _build_structure(self, delta_dict, illegal_bias):

        mask = torch.full((self.S, self.A, self.S), illegal_bias)

        prior = torch.zeros(self.S, self.A, self.S)

        for q in range(self.S):
            if q not in delta_dict:
                continue

            for a, next_states in delta_dict[q].items():
                if len(next_states) == 0:
                    continue

                uniform = 1.0 / len(next_states)

                for q2 in next_states:
                    mask[q, a, q2] = 0.0
                    prior[q, a, q2] = uniform

        return mask, prior

    # -------------------------------------------------
    # 转移概率
    # -------------------------------------------------
    def transition_prob(self, states, actions):
        """
        states:  [B]
        actions: [B]
        return:  [B, S]
        """

        logits = self.theta[states, actions] + self.mask[states, actions]
        return F.softmax(logits, dim=-1)

    # -------------------------------------------------
    # KL结构保持
    # -------------------------------------------------
    def kl_structure_loss(self, states, actions):

        prob = self.transition_prob(states, actions)
        prior = self.prior[states, actions]

        eps = 1e-8

        kl = prior * (torch.log(prior + eps) - torch.log(prob + eps))

        return kl.sum(dim=-1).mean()

    # -------------------------------------------------
    # 熵正则
    # -------------------------------------------------
    def entropy_loss(self, states, actions):

        prob = self.transition_prob(states, actions)

        eps = 1e-8

        entropy = -torch.sum(prob * torch.log(prob + eps), dim=-1)

        return entropy.mean()

    # -------------------------------------------------
    # 总损失
    # -------------------------------------------------
    def total_loss(self, states, actions, task_loss):

        kl_loss = self.kl_structure_loss(states, actions)
        entropy = self.entropy_loss(states, actions)

        total = task_loss + self.kl_lambda * kl_loss - self.entropy_coef * entropy

        return total, {
            "task": task_loss.item(),
            "kl": kl_loss.item(),
            "entropy": entropy.item(),
        }

    # -------------------------------------------------
    # 状态分布传播（IRL常用）
    # -------------------------------------------------
    def propagate(self, state_dist, actions):
        """
        state_dist: [B, S]
        actions: [B]
        """

        B = state_dist.shape[0]
        next_dist = torch.zeros_like(state_dist)

        for q in range(self.S):
            prob = self.transition_prob(
                torch.full((B,), q, device=state_dist.device), actions
            )

            next_dist += state_dist[:, q : q + 1] * prob

        return next_dist

    def action_sequence_loss(self, traj_actions, init_dist):
        """
        traj_actions : [B,T]
        init_dist : [B,S]
        """
        B, T = traj_actions.shape
        state_dist = init_dist
        for t in range(T):
            a = traj_actions[:, t]
            state_dist = self.propagate(state_dist, a)
        # 轨迹概率
        traj_prob = state_dist.sum(dim=-1)
        loss = -torch.log(traj_prob + 1e-8)
        return loss.mean()


# =====================================================
# 数据增强
# =====================================================
def augment_sequence(seq, action_names, prob_insert=0.1, prob_delete=0.2):
    """
    对动作序列做增强：Gu 插入、Gu 删除
    seq: list of action names (str)
    action_names: list of all action names (for index lookup)
    """
    GU = "Gu"
    out = list(seq)

    # Gu 删除：以 prob_delete 概率删除每个 Gu
    out = [a for a in out if not (a == GU and random.random() < prob_delete)]

    # Gu 插入：随机选取若干位置插入 Gu
    n_insert = int(len(out) * prob_insert)
    for _ in range(n_insert):
        pos = random.randint(0, len(out))
        out.insert(pos, GU)

    return out if len(out) > 0 else list(seq)


# =====================================================
# 训练流程
# =====================================================
def _compute_traj_loss(model, traj_actions, init_dist):
    """
    单条轨迹的前向与损失计算，不 backward
    traj_actions : [B,T]
    返回: (total_loss, {task, kl, entropy})
    """
    B, T = traj_actions.shape
    S = init_dist.shape[1]
    state_dist = init_dist.to(traj_actions.device)
    eps = 1e-8

    task_loss = 0.0
    kl_loss = 0.0
    entropy_loss = 0.0

    for t in range(T):
        a = traj_actions[:, t]
        all_prob = []
        for q in range(S):
            states = torch.full((B,), q, device=traj_actions.device, dtype=torch.long)
            prob = model.transition_prob(states, a)
            all_prob.append(prob.unsqueeze(1))
        all_prob = torch.cat(all_prob, dim=1)

        next_dist = torch.einsum("bq,bqs->bs", state_dist, all_prob)

        step_prob = next_dist[:, list(model.accept_states)]
        if step_prob.sum() > 0.5 or t == T - 1:
            task_loss = task_loss + (-torch.log(step_prob + eps).mean())

        prior = model.prior[:, a, :].permute(1, 0, 2)
        kl = prior * (torch.log(prior + eps) - torch.log(all_prob + eps))
        kl = (state_dist.unsqueeze(-1) * kl).sum(dim=(1, 2))
        kl_loss = kl_loss + kl.mean()

        entropy = -all_prob * torch.log(all_prob + eps)
        entropy = (state_dist.unsqueeze(-1) * entropy).sum(dim=(1, 2))
        entropy_loss = entropy_loss + entropy.mean()

        state_dist = next_dist

    task_loss = task_loss / T
    kl_loss = kl_loss / T
    entropy_loss = entropy_loss / T
    total_loss = (
        task_loss + model.kl_lambda * kl_loss - model.entropy_coef * entropy_loss
    )

    return total_loss, {
        "task": task_loss.item(),
        "kl": kl_loss.item(),
        "entropy": entropy_loss.item(),
    }


def train_step(model, optimizer, traj_actions, init_dist):
    """单条轨迹训练（向后兼容）"""
    total_loss, info = _compute_traj_loss(model, traj_actions, init_dist)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    info["total"] = total_loss.item()
    return info


def train_batch_step(model, optimizer, traj_batch, init_dist):
    """
    批量训练：多条序列的 loss 取平均后 backward
    traj_batch: list of tensors [1, T_i]
    """
    losses = []
    infos = []
    for traj in traj_batch:
        loss, info = _compute_traj_loss(model, traj, init_dist)
        losses.append(loss)
        infos.append(info)
    avg_loss = sum(losses) / len(losses)

    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    avg_info = {
        "total": avg_loss.item(),
        "task": sum(i["task"] for i in infos) / len(infos),
        "kl": sum(i["kl"] for i in infos) / len(infos),
        "entropy": sum(i["entropy"] for i in infos) / len(infos),
    }
    return avg_info


def evaluate(model, traj_actions_list, init_dist):
    """
    在验证集上评估，不更新梯度
    traj_actions_list: 验证轨迹列表，每项为 [1,T] 的 tensor
    """
    model.eval()
    eps = 1e-8
    task_loss = 0.0

    with torch.no_grad():
        for traj_actions in traj_actions_list:
            B, T = traj_actions.shape
            S = init_dist.shape[1]
            state_dist = init_dist.to(traj_actions.device)

            for t in range(T):
                a = traj_actions[:, t]
                all_prob = []
                for q in range(S):
                    states = torch.full(
                        (B,), q, device=traj_actions.device, dtype=torch.long
                    )
                    prob = model.transition_prob(states, a)
                    all_prob.append(prob.unsqueeze(1))
                all_prob = torch.cat(all_prob, dim=1)
                next_dist = torch.einsum("bq,bqs->bs", state_dist, all_prob)
                step_prob = next_dist[:, list(model.accept_states)]
                if step_prob.sum() > 0.5 or t == T - 1:
                    task_loss = task_loss + (-torch.log(step_prob + eps).mean())
                state_dist = next_dist

            task_loss = task_loss / T

    model.train()
    return {
        "val_task": task_loss,
    }


# =====================================================
# 示例运行
# =====================================================
def print_transition_table(model, state_names, action_names):
    """
    打印训练完成后的转移概率表

    model         : 你的 NPA 模型
    state_names   : 状态名称列表
    action_names  : 动作名称列表
    """

    S = len(state_names)
    A = len(action_names)

    with torch.no_grad():
        prob = F.softmax(model.theta + model.mask, dim=-1)
        # prob shape = [S, A, S]

        for a in range(A):
            print("\n")
            print(f"====== Action: {action_names[a]} ======")

            table = prob[:, a, :].cpu().numpy()

            df = pd.DataFrame(table, index=state_names, columns=state_names)

            print(df.round(4))


if __name__ == "__main__":
    alphabet_to_index = {a: i for i, a in enumerate(sorted(alphabet))}
    state_to_index = {s: i for i, s in enumerate(sorted(states))}

    accept_states = {"q8"}
    delta = {
        state_to_index[q]: {
            alphabet_to_index[a]: {state_to_index[q2] for q2 in next_states}
            for a, next_states in a_dict.items()
        }
        for q, a_dict in transitions.items()
    }
    accept_states = {state_to_index[q] for q in accept_states}
    num_states = len(states)
    num_actions = len(alphabet)

    expert_seqs = get_expert_seqs()
    action_names = list(sorted(alphabet))

    def seq_to_tensor(seq):
        return torch.tensor([[alphabet_to_index[a] for a in seq]], dtype=torch.long)

    traj_actions_list = [seq_to_tensor(seq) for seq in expert_seqs]

    # ---------- 划分训练集/验证集（5条为验证） ----------
    random.seed(int(time.time()))
    indices = list(range(len(expert_seqs)))
    random.shuffle(indices)
    val_indices = indices[:5]
    train_indices = indices[5:]

    train_seqs = [expert_seqs[i] for i in train_indices]
    val_traj_list = [traj_actions_list[i] for i in val_indices]

    init_dist = torch.tensor([[1.0] + [0.0] * (num_states - 1)])

    # ---------- 强化正则化（缓解序列间相关性差） ----------
    model = NPA(
        num_states=num_states,
        num_actions=num_actions,
        delta_dict=delta,
        accept_states=accept_states,
        kl_lambda=1.5,
        entropy_coef=0.15,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 8
    aug_prob_insert = 0.1
    aug_prob_delete = 0.2

    # ---------- 训练：批量 + 数据增强 ----------
    for epoch in range(10):
        train_traj_list = []
        for seq in train_seqs:
            if random.random() < 0.5:
                seq = augment_sequence(
                    seq, action_names, aug_prob_insert, aug_prob_delete
                )
            train_traj_list.append(seq_to_tensor(seq))

        n_steps = (len(train_traj_list) + batch_size - 1) // batch_size
        for step in range(n_steps):
            batch_indices = random.sample(
                range(len(train_traj_list)), min(batch_size, len(train_traj_list))
            )
            traj_batch = [train_traj_list[i] for i in batch_indices]
            info = train_batch_step(model, optimizer, traj_batch, init_dist)
            if step % 5 == 0:
                print(f"Epoch {epoch} step {step}", info)

        val_info = evaluate(model, val_traj_list, init_dist)
        print(f"Epoch {epoch} val: {val_info}")

    # ---------- 保存模型到 model/ ----------
    os.makedirs("model", exist_ok=True)
    save_path = os.path.join("model", "npa.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_states": num_states,
            "num_actions": num_actions,
            "accept_states": list(accept_states),
        },
        save_path,
    )
    print(f"Model saved to {save_path}")
    print_transition_table(model, list(states), list(sorted(alphabet)))
