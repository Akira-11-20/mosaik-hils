"""
カスタムプロットユーティリティ

mosaikの標準プロットを拡張し、データのやり取りがあるステップのみを表示する機能を提供。
"""

from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional

try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import MaxNLocator
    from matplotlib.patches import ConnectionPatch
    import networkx as nx

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
            dot = ax.plot(
                steps_st[sim_name], [i] * len(steps_st[sim_name]), "o", markersize=3
            )
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
        print(f"   📊 Full execution graph saved")
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
        node_color: ノードの色
        node_alpha: ノードの透明度
        label_alpha: ラベルの透明度
        edge_alpha: エッジの透明度
        arrow_size: 矢印のサイズ
        figsize: 図のサイズ (width, height)
        exclude_nodes: 除外するノード名のリスト（部分一致）
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

    # プロット設定
    fig, ax = plt.subplots(figsize=figsize)

    # ノード描画
    for node in df_graph.nodes:
        ax.plot(
            positions[node][0],
            positions[node][1],
            "o",
            markersize=node_size / 10,  # matplotlibのマーカーサイズ調整
            color=node_color,
            alpha=node_alpha,
        )

        # ラベル配置
        text_x = positions[node][0]
        text_y = positions[node][1]
        label = ax.annotate(
            node,
            positions[node],
            xytext=(text_x, text_y),
            size=node_label_size,
            ha="center",
            va="center",
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
