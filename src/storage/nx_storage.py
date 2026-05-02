import networkx as nx
import os
from datetime import datetime
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)



def save_graph(G: nx.Graph, filepath: str, format: str = "graphml"):
    """
    保存图为文件

    Args:
        G: NetworkX 图
        filepath: 输出路径
        format: 保存格式 (graphml, gexf, gml)
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if format == "graphml":
        nx.write_graphml(G, filepath)
    elif format == "gexf":
        nx.write_gexf(G, filepath)
    elif format == "gml":
        nx.write_gml(G, filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_graph(filepath: str, format: str = "graphml") -> nx.Graph:
    """
    从文件加载图

    Args:
        filepath: 图文件路径
        format: 文件格式

    Returns:
        NetworkX 图
    """
    if format == "graphml":
        return nx.read_graphml(filepath)
    elif format == "gexf":
        return nx.read_gexf(filepath)
    elif format == "gml":
        return nx.read_gml(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def get_graph_info(G: nx.Graph) -> dict:
    """
    获取图的基本信息

    Args:
        G: NetworkX 图

    Returns:
        图信息字典
    """
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "is_connected": nx.is_connected(G),
        "density": nx.density(G),
        "average_clustering": nx.average_clustering(G),
    }


def export_to_json(G: nx.Graph, filepath: str):
    """
    导出图为 JSON 格式 (可用于前端可视化)

    Args:
        G: NetworkX 图
        filepath: 输出路径
    """
    import json

    nodes = []
    for node in G.nodes(data=True):
        nodes.append({
            "id": node[0],
            **node[1]
        })

    edges = []
    for edge in G.edges(data=True):
        edges.append({
            "source": edge[0],
            "target": edge[1],
            **edge[2]
        })

    data = {
        "generated_at": datetime.now().isoformat(),
        "info": get_graph_info(G),
        "nodes": nodes,
        "edges": edges
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)