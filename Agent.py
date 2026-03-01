"""
课堂对话 Agent：用提示词引导第一句发言，再根据上下文逐句生成下一句发言。
输出完整对话列表。

依赖：pip install openai python-dotenv
配置：.env 中设置 OPENAI_API_KEY、OPENAI_BASE_URL（或传入 api_key / base_url）
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 发言风格约定（供提示词使用）
PROMPT_STYLE_GUIDE = """
【开场】真实课堂常以「老师：上课。」「学生：起立，老师好。」开场；随后老师可问预习、引入课题等。
【发言风格】学生可极简短（如「对。」「1。」「不全面。」）；老师可点名「你来说」「谁来补充」；内容需贴合当前科目与课堂主题。
【结束】课堂结束时常为老师布置作业或说「下课」、学生「起立，老师再见」。
"""


def _create_client(
    api_key: Optional[str] = None, base_url: Optional[str] = None
) -> OpenAI:
    """创建 OpenAI 兼容客户端。从 .env 读取 OPENAI_API_KEY、OPENAI_BASE_URL。"""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# 1. 使用提示词引导出第一次发言
# ---------------------------------------------------------------------------
def _build_first_utterance_prompt(topic: str, subject: str, grade: str) -> str:
    return f"""你正在模拟一堂课的对话流程。请根据以下课堂信息，生成老师的**第一句发言**。

- 课堂主题：{topic}
- 科目：{subject}
- 年级：{grade}
{PROMPT_STYLE_GUIDE}

要求：
- 只生成**一条**发言，格式严格为：老师：发言内容
- 若模拟完整上课流程，第一句通常为「老师：上课。」；若直接进入主题，可为「老师：同学们，……」（如问预习、引入课题等）。
- 不要换行、不要解释。"""


def get_first_utterance(
    topic: str,
    subject: str,
    grade: str,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """使用提示词（含课堂主题、科目、年级）引导 LLM 生成第一句发言。"""
    client = client or _create_client()
    prompt = _build_first_utterance_prompt(topic, subject, grade)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    text = (resp.choices[0].message.content or "").strip()
    if "老师：" in text:
        line = text.split("老师：")[-1].split("学生：")[0].strip()
        return f"老师：{line}" if line else "老师：上课。"
    return f"老师：{text[:80]}" if text else "老师：上课。"


# ---------------------------------------------------------------------------
# 2. 根据上下文输出下一次发言
# ---------------------------------------------------------------------------
def _build_next_utterance_prompt(
    dialogue_so_far: list[str],
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
    return f"""根据当前对话，生成**下一条**发言（老师或学生的一方）。
{course_line}
已有对话：
{ctx}
{PROMPT_STYLE_GUIDE}

要求：
- 只生成**一条**发言，格式严格为：老师：发言内容 或 学生：发言内容（由上下文决定该谁说话、说什么）。
- 发言风格：学生可极简短；老师可点名；内容贴合当前科目与主题。
- 不要换行、不要多条，不要解释。"""


def get_next_utterance(
    dialogue_so_far: list[str],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
    topic: str = "",
    subject: str = "",
    grade: str = "",
) -> str:
    """根据上下文（已有对话）输出下一次发言。"""
    client = client or _create_client()
    prompt = _build_next_utterance_prompt(
        dialogue_so_far, topic=topic, subject=subject, grade=grade
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    text = (resp.choices[0].message.content or "").strip()
    if "老师：" in text:
        line = text.split("老师：")[-1].split("学生：")[0].strip()
        return f"老师：{line}" if line else "老师：请继续。"
    if "学生：" in text:
        line = text.split("学生：")[-1].split("老师：")[0].strip()
        return f"学生：{line}" if line else "学生：好的。"
    return f"学生：{text[:80]}" if text else "学生：好的。"


# ---------------------------------------------------------------------------
# 3. 运行一轮对话：第一句 + 循环「根据上下文生成下一句」
# ---------------------------------------------------------------------------
def run_session(
    topic: str,
    subject: str,
    grade: str,
    max_turns: int = 30,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """
    运行一轮课堂对话：用提示词引导第一句，再根据上下文逐句生成下一句。
    返回：对话列表 ["老师：…", "学生：…", ...]
    """
    client = client or _create_client()
    dialogue: list[str] = []

    first_utterance = get_first_utterance(
        topic, subject, grade, client=client, model=model
    )
    dialogue.append(first_utterance)

    for _ in range(max_turns - 1):
        utterance = get_next_utterance(
            dialogue,
            client=client,
            model=model,
            topic=topic,
            subject=subject,
            grade=grade,
        )
        dialogue.append(utterance)
        if "下课" in utterance or "再见" in utterance:
            break

    return dialogue


# ---------------------------------------------------------------------------
# 批量生成非专家轨迹并保存到 non_expert/ 目录
# ---------------------------------------------------------------------------
def _generate_one_non_expert(
    i: int,
    n: int,
    output_dir: str,
    topic: str,
    subject: str,
    grade: str,
    min_len: int,
    max_len: int,
    max_turns_per_session: int,
    model_agent: str,
    model_judger: str,
):
    """
    单条非专家轨迹生成（供多线程调用）。每线程使用独立 client 避免并发冲突。
    返回 (i, True, path, len_actions) 或 (i, False, None, error_msg)。
    """
    from Judger import generate_non_expert_trajectory_by_agent

    client = _create_client()
    try:
        result = generate_non_expert_trajectory_by_agent(
            topic=topic,
            subject=subject,
            grade=grade,
            min_len=min_len,
            max_len=max_len,
            max_turns_per_session=max_turns_per_session,
            client_agent=client,
            client_judger=client,
            model_agent=model_agent,
            model_judger=model_judger,
        )
    except Exception as e:
        return (i, False, None, str(e))
    if result is None:
        return (i, False, None, "轨迹过短")
    actions, dialogue = result
    path = os.path.join(output_dir, f"{i:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"dialogue": dialogue, "trajectory": actions},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return (i, True, path, len(actions))


def batch_generate_non_expert_trajectories(
    n: int = 50,
    output_dir: str = "non_expert",
    min_len: int = 100,
    max_len: int = 150,
    max_turns_per_session: int = 30,
    model_agent: str = "gpt-4o-mini",
    model_judger: str = "gpt-4o",
    client=None,
    max_workers: int = 5,
) -> int:
    """
    批量生成 n 条非专家轨迹（多线程）：每条由 Agent 生成对话，Judger 标注动作。
    保存到 output_dir/001.json ... output_dir/{n:03d}.json。
    max_workers: 线程数，默认 5；设为 1 则退化为单线程。
    返回成功保存的条数。
    """
    os.makedirs(output_dir, exist_ok=True)
    presets = [
        ("我与地坛", "语文", "高中一年级"),
        ("一元二次方程的解法", "数学", "初中二年级"),
        ("牛顿第一定律", "物理", "高中一年级"),
        ("细胞的结构与功能", "生物", "初中一年级"),
        ("辛亥革命", "历史", "高中一年级"),
    ]
    saved = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(1, n + 1):
            topic, subject, grade = presets[(i - 1) % len(presets)]
            fut = executor.submit(
                _generate_one_non_expert,
                i,
                n,
                output_dir,
                topic,
                subject,
                grade,
                min_len,
                max_len,
                max_turns_per_session,
                model_agent,
                model_judger,
            )
            futures[fut] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                i_out, ok, path, extra = fut.result()
            except Exception as e:
                print(f"[{i}/{n}] 异常: {e}")
                continue
            if ok:
                saved += 1
                print(f"[{i_out}/{n}] 已保存 {path} (轨迹长度 {extra})")
            else:
                print(f"[{i_out}/{n}] 失败 ({extra})")
    return saved


# ---------------------------------------------------------------------------
# 命令行示例
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        n = 50
        if len(sys.argv) > 2:
            try:
                n = int(sys.argv[2])
            except ValueError:
                pass
        print(f"批量生成 {n} 条非专家轨迹到 non_expert/ ...")
        saved = batch_generate_non_expert_trajectories(n=n, output_dir="non_expert")
        print(f"完成，共保存 {saved} 条")
        return


if __name__ == "__main__":
    main()
