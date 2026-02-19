from graphviz import Digraph
from AutomatonStruct import states, transitions, start_state, accept_states


# 创建一个有向图（graphviz Digraph）
dot = Digraph(strict=True, format="png")

# 设置图的属性，使得布局更加清晰
dot.attr(dpi="300", size="10,10", rankdir="LR", fontname="Helvetica", fontsize="12")

# 添加节点
for state in states:
    if state in accept_states:
        dot.node(
            state,
            shape="doublecircle",
            color="green",
            style="filled",
            fontcolor="black",
        )  # 接受状态使用双圈和绿色
    elif state == start_state:
        dot.node(
            state, shape="circle", color="red", style="filled", fontcolor="white"
        )  # 起始状态使用红色
    else:
        dot.node(state, shape="circle")

# 添加转移边
for state, transitions_dict in transitions.items():
    for symbol, next_states in transitions_dict.items():
        for next_state in next_states:
            dot.edge(state, next_state, label=symbol)

# 渲染图形并保存为 PNG
dot.render("nfa_graph", view=True)
