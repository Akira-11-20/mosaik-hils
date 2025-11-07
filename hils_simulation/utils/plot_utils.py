"""
カスタムプロットユーティリティ

mosaikの標準プロットを拡張し、データのやり取りがあるステップのみを表示する機能を提供。
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib import rcParams
    from matplotlib.patches import ConnectionPatch
    from matplotlib.ticker import MaxNLocator

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_execution_graph_with_data_only(
    world,
    title: str = "",
    folder: str = "figures",
    dpi: int = 600,
    format: str = "png",
    show_plot: bool = True,
    save_plot: bool = True,
):
    """
    データのやり取りがあったステップのみを表示する実行グラフを作成

    Args:
        world: mosaik world object
        title: グラフのタイトル
        folder: 出力フォルダ
        dpi: 画像解像度
        format: 画像フォーマット (png, pdf, svg)
        show_plot: プロット表示するか
        save_plot: ファイルに保存するか
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not available, skipping plot")
        return

    if not hasattr(world, "execution_graph"):
        print("⚠️  No execution graph available (debug mode required)")
        return

    # 全ノードとエッジを取得
    all_nodes = list(world.execution_graph.nodes(data=True))
    all_edges = list(world.execution_graph.edges())

    if not all_edges:
        print("⚠️  No edges in execution graph")
        return

    # データのやり取りがあったノードを抽出
    nodes_with_data = set()
    for edge in all_edges:
        nodes_with_data.add(edge[0])
        nodes_with_data.add(edge[1])

    # シミュレーターごとにステップ位置を記録
    steps_st: Dict[str, List[float]] = {}
    for sim_name in world.sims.keys():
        steps_st[sim_name] = []

    # データのやり取りがあったノードのみ記録
    for node in all_nodes:
        if node[0] in nodes_with_data:
            sim_name, tiered_time = node[0]
            # tiered_timeから時刻を取得
            time_pos = tiered_time.time
            steps_st[sim_name].append(time_pos)

    # プロット設定
    rcParams.update({"figure.autolayout": True})
    fig, ax = plt.subplots(figsize=(12, 6))

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("Execution Graph (Data Exchanges Only)")

    # シミュレーターごとにプロット
    colormap = ["black" for _ in world.sims]
    for i, sim_name in enumerate(world.sims):
        if steps_st[sim_name]:  # データがある場合のみプロット
            dot = ax.plot(steps_st[sim_name], [i] * len(steps_st[sim_name]), "o", markersize=3)
            colormap[i] = dot[0].get_color()

    # 軸設定
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticks(list(range(len(world.sims.keys()))))
    ax.set_yticklabels(list(world.sims.keys()))
    ax.set_xlabel("Simulation Time [ms]")
    ax.set_ylabel("Simulator")

    # Y軸位置のマッピング
    y_pos: Dict[str, int] = {}
    for sim_count, sim_name in enumerate(world.sims.keys()):
        y_pos[sim_name] = sim_count

    # エッジ（データフロー）を描画
    for edge in all_edges:
        isid_0, t0 = edge[0]
        isid_1, t1 = edge[1]

        x_pos0 = t0.time
        x_pos1 = t1.time
        y_pos0 = y_pos[isid_0]
        y_pos1 = y_pos[isid_1]

        # 矢印で接続を表示
        ax.annotate(
            "",
            xy=(x_pos1, y_pos1),
            xytext=(x_pos0, y_pos0),
            arrowprops=dict(arrowstyle="->", color=colormap[y_pos0], alpha=0.3, lw=0.5),
        )

    # グリッド表示
    ax.grid(True, alpha=0.3)

    # 保存とプロット表示
    if save_plot:
        output_dir = Path(folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"execution_graph_data_only.{format}"
        plt.savefig(output_path, dpi=dpi, format=format)
        print(f"   📊 Execution graph (data only) saved: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_execution_graph_comparison(
    world,
    folder: str = "figures",
    dpi: int = 600,
    format: str = "png",
    show_plot: bool = False,
):
    """
    標準の実行グラフとデータのみの実行グラフを両方生成

    Args:
        world: mosaik world object
        folder: 出力フォルダ
        dpi: 画像解像度
        format: 画像フォーマット
        show_plot: プロット表示するか
    """
    # 標準の実行グラフ（mosaik.util）
    try:
        import mosaik.util

        mosaik.util.plot_execution_graph(
            world,
            title="Full Execution Graph",
            folder=folder,
            dpi=dpi,
            format=format,
            show_plot=show_plot,
            save_plot=True,
        )
        print("   📊 Full execution graph saved")
    except Exception as e:
        print(f"   ⚠️  Full execution graph failed: {e}")

    # データのみの実行グラフ（カスタム）
    plot_execution_graph_with_data_only(
        world,
        title="Execution Graph (Data Exchanges Only)",
        folder=folder,
        dpi=dpi,
        format=format,
        show_plot=show_plot,
        save_plot=True,
    )


def plot_dataflow_graph_custom(
    world,
    folder: str = "figures",
    hdf5path: Optional[str] = None,
    dpi: int = 600,
    format: Literal["png", "pdf", "svg"] = "png",
    show_plot: bool = True,
    # カスタマイズ可能なパラメータ
    node_size: int = 100,
    node_label_size: int = 8,
    edge_label_size: int = 6,
    node_color: str = "tab:blue",
    node_alpha: float = 0.8,
    label_alpha: float = 0.6,
    edge_alpha: float = 0.6,
    arrow_size: int = 20,
    figsize: Tuple[float, float] = (10, 8),
    exclude_nodes: Optional[List[str]] = None,
    label_position: Literal["center", "top", "bottom", "left", "right", "auto"] = "auto",
    color_by_type: bool = True,
):
    """
    データフローグラフをカスタマイズ可能なパラメータで描画

    mosaikの標準plot_dataflow_graphをベースに、ノードサイズや
    ラベルサイズなどをカスタマイズできるように拡張。

    Args:
        world: mosaik world object
        folder: 出力フォルダ
        hdf5path: HDF5ファイルパス（指定時はそのパスを使用）
        dpi: 画像解像度
        format: 画像フォーマット (png, pdf, svg)
        show_plot: プロット表示するか
        node_size: ノードのサイズ
        node_label_size: ノードラベルのフォントサイズ
        edge_label_size: エッジラベルのフォントサイズ
        node_color: ノードの色（color_by_type=Falseの場合に使用）
        node_alpha: ノードの透明度
        label_alpha: ラベルの透明度
        edge_alpha: エッジの透明度
        arrow_size: 矢印のサイズ
        figsize: 図のサイズ (width, height)
        exclude_nodes: 除外するノード名のリスト（部分一致）
        label_position: ラベルの配置位置（center/top/bottom/left/right/auto）
        color_by_type: ノード種別ごとに色分けするか
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not available, skipping plot")
        return

    # 除外ノードの設定
    if exclude_nodes is None:
        exclude_nodes = []

    # データフローグラフを再構築
    df_graph: nx.DiGraph = nx.DiGraph()
    for sim in world.sims.values():
        # 除外チェック（部分一致）
        should_exclude = False
        for exclude_pattern in exclude_nodes:
            if exclude_pattern in sim.sid:
                should_exclude = True
                break

        if should_exclude:
            continue

        df_graph.add_node(sim.sid)
        for pred, delay in sim.input_delays.items():
            # 前ノードも除外されていないかチェック
            pred_excluded = False
            for exclude_pattern in exclude_nodes:
                if exclude_pattern in pred.sid:
                    pred_excluded = True
                    break

            if pred_excluded:
                continue

            df_graph.add_edge(
                pred.sid,
                sim.sid,
                time_shifted=delay.is_time_shifted(),
                weak=delay.is_weak(),
            )

    # レイアウト計算（Fruchterman-Reingold force-directed algorithm）
    positions = nx.spring_layout(df_graph)

    # ノード種別の判定と色マッピング
    def get_node_type(node_name: str) -> str:
        """ノード名からシミュレータータイプを判定"""
        if "Controller" in node_name:
            return "Controller"
        elif "Plant" in node_name:
            return "Plant"
        elif "Env" in node_name or "Spacecraft" in node_name:
            return "Environment"
        elif "Bridge" in node_name or "Comm" in node_name:
            return "Bridge"
        elif "Collector" in node_name or "Data" in node_name:
            return "Collector"
        elif "InverseComp" in node_name:
            return "InverseComp"
        else:
            return "Other"

    # 色のマッピング（落ち着いたトーン）
    type_colors = {
        "Controller": "#8B7D8B",  # 落ち着いた紫グレー
        "Plant": "#6B8E8F",  # 落ち着いた青緑
        "Environment": "#7A9EAB",  # 落ち着いた青
        "Bridge": "#B59F7B",  # 落ち着いたベージュ
        "Collector": "#87A886",  # 落ち着いた緑
        "InverseComp": "#9B8FA3",  # 落ち着いた薄紫
        "Other": "#9B9B9B",  # グレー
    }

    # ラベル位置のオフセット計算
    def calculate_label_offset(
        pos_x: float, pos_y: float, all_positions: dict, position_mode: str
    ) -> Tuple[float, float]:
        """ラベルのオフセット位置を計算（ノードと重ならないように）"""
        offset_distance = 0.15  # ノードからの距離

        if position_mode == "center":
            return (pos_x, pos_y)
        elif position_mode == "top":
            return (pos_x, pos_y + offset_distance)
        elif position_mode == "bottom":
            return (pos_x, pos_y - offset_distance)
        elif position_mode == "left":
            return (pos_x - offset_distance, pos_y)
        elif position_mode == "right":
            return (pos_x + offset_distance, pos_y)
        elif position_mode == "auto":
            # 自動配置: 他のノードとの距離を考慮して最適な位置を選択
            candidates = [
                ("top", pos_x, pos_y + offset_distance),
                ("bottom", pos_x, pos_y - offset_distance),
                ("left", pos_x - offset_distance, pos_y),
                ("right", pos_x + offset_distance, pos_y),
            ]

            # 他のノードとの距離が最大になる位置を選択
            best_pos = candidates[0]
            max_min_dist = 0

            for direction, cx, cy in candidates:
                min_dist = float("inf")
                for other_pos in all_positions.values():
                    dist = ((cx - other_pos[0]) ** 2 + (cy - other_pos[1]) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_pos = (direction, cx, cy)

            return (best_pos[1], best_pos[2])
        else:
            return (pos_x, pos_y + offset_distance)  # デフォルトは上

    # プロット設定
    fig, ax = plt.subplots(figsize=figsize)

    # ノード描画
    for node in df_graph.nodes:
        # ノードタイプの判定
        node_type = get_node_type(node)

        # 色の決定
        if color_by_type:
            current_color = type_colors.get(node_type, node_color)
        else:
            current_color = node_color

        ax.plot(
            positions[node][0],
            positions[node][1],
            "o",
            markersize=node_size / 10,  # matplotlibのマーカーサイズ調整
            color=current_color,
            alpha=node_alpha,
        )

        # ラベル配置（ノードの外側）
        label_x, label_y = calculate_label_offset(positions[node][0], positions[node][1], positions, label_position)

        # 水平・垂直配置の調整
        if label_position == "center":
            h_align, v_align = "center", "center"
        elif label_position in ["top", "auto"] or (label_position == "auto" and label_y > positions[node][1]):
            h_align, v_align = "center", "bottom"
        elif label_position == "bottom" or (label_position == "auto" and label_y < positions[node][1]):
            h_align, v_align = "center", "top"
        elif label_position == "left" or (label_position == "auto" and label_x < positions[node][0]):
            h_align, v_align = "right", "center"
        elif label_position == "right" or (label_position == "auto" and label_x > positions[node][0]):
            h_align, v_align = "left", "center"
        else:
            h_align, v_align = "center", "bottom"

        label = ax.annotate(
            node,
            positions[node],
            xytext=(label_x, label_y),
            size=node_label_size,
            ha=h_align,
            va=v_align,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7),
        )
        label.set_alpha(label_alpha)

    # エッジ描画
    for edge in list(df_graph.edges()):
        edge_infos = df_graph.adj[edge[0]][edge[1]]
        annotation = ""
        color = "grey"
        linestyle = "solid"

        if edge_infos["time_shifted"]:
            color = "tab:red"
            annotation = "time_shifted"

        if edge_infos["weak"]:
            annotation += " weak" if annotation else "weak"
            linestyle = "dotted"

        x_pos0 = positions[edge[0]][0]
        x_pos1 = positions[edge[1]][0]
        y_pos0 = positions[edge[0]][1]
        y_pos1 = positions[edge[1]][1]

        con = ConnectionPatch(
            (x_pos0, y_pos0),
            (x_pos1, y_pos1),
            "data",
            "data",
            arrowstyle="->",
            linestyle=linestyle,
            connectionstyle="arc3,rad=0.1",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=arrow_size,
            fc="w",
            color=color,
            alpha=edge_alpha,
        )
        ax.add_artist(con)

        # エッジラベル（アノテーション）
        if annotation:
            midpoint: Tuple[float, float] = con.get_path().vertices[1]
            ax.annotate(
                annotation,
                (midpoint[0], midpoint[1]),
                xytext=(0, 0),
                textcoords="offset points",
                color=color,
                fontsize=edge_label_size,
                alpha=edge_alpha,
            )

    plt.axis("off")

    # プロット表示
    if show_plot:
        plt.show()

    # ファイル保存
    if hdf5path:
        filename = hdf5path.replace(".hdf5", f"graph_df.{format}")
    else:
        output_dir = Path(folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = str(output_dir / f"dataflowGraph_custom.{format}")

    fig.savefig(
        filename,
        format=format,
        dpi=dpi,
        facecolor="white",
        transparent=True,
        bbox_inches="tight",
    )

    print(f"   📊 Custom dataflow graph saved: {filename}")

    if not show_plot:
        plt.close()
