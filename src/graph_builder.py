"""
药食同源饮品辅助配方推荐系统 —— 知识图谱构建模块。

本模块不强制依赖图数据库，优先使用轻量级字典结构表达：
疾病 -> 靶点 -> 成分 -> 原料
同时提供边列表，便于后续接入 networkx、可视化或图算法。
"""

from typing import Dict, List, Optional, Set, Tuple

from src.data_loader import DataLoader


class KnowledgeGraphBuilder:
    """基于 DataLoader 中的标准化数据构建轻量知识图谱。"""

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Args:
            data_loader: 已加载数据的 DataLoader；若为空则自动加载 data/data.xlsx。
        """
        self.data = data_loader if data_loader is not None else DataLoader()
        self.nodes: Dict[str, Set[str]] = {
            "疾病": set(),
            "靶点": set(),
            "成分": set(),
            "原料": set(),
        }
        self.edges: List[Dict] = []
        self.adjacency: Dict[str, Dict[str, List[str]]] = {}

    def build(self) -> Dict:
        """
        构建知识图谱并返回完整结构。

        Returns:
            {
                "nodes": {"疾病": [...], "靶点": [...], "成分": [...], "原料": [...]},
                "edges": [{"source": str, "target": str, "relation": str, ...}, ...],
                "adjacency": {source: {relation: [target, ...]}}
            }
        """
        self.nodes = {"疾病": set(), "靶点": set(), "成分": set(), "原料": set()}
        self.edges = []
        self.adjacency = {}

        self._add_disease_target_edges()
        self._add_target_component_edges()
        self._add_component_substance_edges()
        self._add_synergy_edges()

        return {
            "nodes": {node_type: sorted(values) for node_type, values in self.nodes.items()},
            "edges": self.edges,
            "adjacency": self.adjacency,
        }

    def _add_edge(self, source: str, target: str, relation: str, **attrs) -> None:
        """添加图谱边，并维护邻接表索引。"""
        self.edges.append({
            "source": source,
            "target": target,
            "relation": relation,
            **attrs,
        })
        self.adjacency.setdefault(source, {}).setdefault(relation, []).append(target)

    def _add_disease_target_edges(self) -> None:
        """添加 疾病 -> 靶点 边。"""
        for disease, targets in self.data.disease_to_targets.items():
            self.nodes["疾病"].add(disease)
            for target in targets:
                self.nodes["靶点"].add(target)
                self._add_edge(disease, target, "作用于靶点")

    def _add_target_component_edges(self) -> None:
        """添加 靶点 -> 成分 边，并保留起效剂量。"""
        for target, components in self.data.target_to_components.items():
            self.nodes["靶点"].add(target)
            for component, dose in components:
                self.nodes["成分"].add(component)
                self._add_edge(target, component, "由成分干预", 起效剂量mg=dose)

    def _add_component_substance_edges(self) -> None:
        """添加 成分 -> 原料 边，并保留含量占比和吸收率。"""
        for component, substances in self.data.component_to_substances.items():
            self.nodes["成分"].add(component)
            for substance, ratio, absorption in substances:
                self.nodes["原料"].add(substance)
                self._add_edge(
                    component,
                    substance,
                    "存在于原料",
                    成分含量占比=ratio,
                    消化吸收率=absorption,
                )

    def _add_synergy_edges(self) -> None:
        """添加 原料 <-> 原料 协同边，去除双向重复记录。"""
        seen: Set[Tuple[str, str]] = set()
        for (sub_a, sub_b), coef in self.data.synergy_map.items():
            key = tuple(sorted((sub_a, sub_b)))
            if key in seen:
                continue
            seen.add(key)
            self.nodes["原料"].update([sub_a, sub_b])
            self._add_edge(sub_a, sub_b, "协同增效", 协同系数=coef)

    def get_graph_summary(self) -> Dict[str, int]:
        """返回节点和边数量摘要，方便前端或调试展示。"""
        graph = self.build()
        return {
            "疾病节点": len(graph["nodes"]["疾病"]),
            "靶点节点": len(graph["nodes"]["靶点"]),
            "成分节点": len(graph["nodes"]["成分"]),
            "原料节点": len(graph["nodes"]["原料"]),
            "关系边": len(graph["edges"]),
        }


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    print(builder.get_graph_summary())
