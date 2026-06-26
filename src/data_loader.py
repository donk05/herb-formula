"""
药食同源饮品辅助配方推荐系统 —— 数据加载模块
使用 pandas 读取 data.xlsx，构建多级字典映射以便快速查询。
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DataLoader:
    """
    数据加载与字典映射构建类。

    负责读取 Excel 中的 4 个 Sheet（Disease, Substance, Mechanism, Synergy），
    并构建以下映射关系以支持高效查询：

    使用示例:
        loader = DataLoader()                      # 自动定位 data/data.xlsx
        loader = DataLoader("/path/to/data.xlsx")  # 或指定路径
        targets = loader.get_targets_by_disease("糖尿病")
        substances = loader.get_substances_by_component("Component_X")
    """

    # 默认数据文件相对于项目根目录的路径
    _DEFAULT_DATA_PATH = "data/data.xlsx"

    def __init__(self, filepath: Optional[str] = None):
        """
        初始化并加载所有数据。

        Args:
            filepath: Excel 文件路径。若为 None，则自动基于本模块所在位置
                      推算项目根目录并拼接 {_DEFAULT_DATA_PATH}。
        """
        if filepath is None:
            # 动态推算项目根目录：本文件在 src/ 下，上级目录即项目根
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            filepath = os.path.join(project_root, self._DEFAULT_DATA_PATH)

        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.filepath}")

        # ---- 原始 DataFrame ----
        self.disease_df: pd.DataFrame = None
        self.substance_df: pd.DataFrame = None
        self.mechanism_df: pd.DataFrame = None
        self.synergy_df: pd.DataFrame = None

        # ---- 核心映射字典 ----
        # 疾病名称 -> [靶点1, 靶点2, ...]
        self.disease_to_targets: Dict[str, List[str]] = {}
        # 作用靶点 -> [(成分1, 起效剂量), ...]
        self.target_to_components: Dict[str, List[Tuple[str, int]]] = {}
        # 包含成分 -> [(物质名称, 成分含量占比, 消化吸收率), ...]
        self.component_to_substances: Dict[str, List[Tuple[str, float, float]]] = {}
        # 物质名称 -> {"成分": str, "含量占比": float, "吸收率": float}
        self.substance_info: Dict[str, Dict] = {}
        # (物质A, 物质B) -> 协同系数（双向均可查询）
        self.synergy_map: Dict[Tuple[str, str], float] = {}

        # 执行加载与构建
        self._load_raw_data()
        self._build_disease_mapping()
        self._build_mechanism_mapping()
        self._build_substance_mapping()
        self._build_synergy_mapping()

    # ===================== 1. 原始数据读取 =====================

    def _load_raw_data(self) -> None:
        """从 Excel 读取所有 Sheet 的原始数据。"""
        xls = pd.ExcelFile(self.filepath)

        # --- Disease：第 0 行为元数据(Column1/Column2)，第 1 行为真实表头，跳过前 2 行 ---
        self.disease_df = pd.read_excel(
            xls, sheet_name="Disease", header=None, skiprows=2
        )
        self.disease_df.columns = ["疾病名称", "作用靶点"]
        # 去除可能的空行
        self.disease_df.dropna(inplace=True)

        # --- Substance：第 0 行为表头 ---
        self.substance_df = pd.read_excel(
            xls, sheet_name="Substance", header=0
        )
        # 统一列名
        self.substance_df.columns = ["物质名称", "包含成分", "成分含量占比", "消化吸收率"]
        self.substance_df.dropna(inplace=True)

        # --- Mechanism：第 0 行为表头 ---
        self.mechanism_df = pd.read_excel(
            xls, sheet_name="Mechanism", header=0
        )
        self.mechanism_df.columns = ["包含成分", "作用靶点", "起效剂量mg"]
        self.mechanism_df.dropna(inplace=True)

        # --- Synergy：第 0 行为表头 ---
        self.synergy_df = pd.read_excel(
            xls, sheet_name="Synergy", header=0
        )
        self.synergy_df.columns = ["物质A", "物质B", "协同系数"]
        self.synergy_df.dropna(inplace=True)

    # ===================== 2. 构建多级映射 =====================

    def _build_disease_mapping(self) -> None:
        """
        构建 疾病 → 靶点 映射。

        映射结果示例:
            {"糖尿病": ["Target_A", "Target_B"],
             "高血压": ["Target_C", "Target_D"]}
        """
        for _, row in self.disease_df.iterrows():
            disease = row["疾病名称"]
            target = row["作用靶点"]
            self.disease_to_targets.setdefault(disease, []).append(target)

    def _build_mechanism_mapping(self) -> None:
        """
        构建 靶点 → (成分, 起效剂量) 映射。

        从 Mechanism 表读取每条记录，按靶点聚合。
        映射结果示例:
            {"Target_A": [("Component_X", 100)],
             "Target_C": [("Component_X", 150)]}
        """
        for _, row in self.mechanism_df.iterrows():
            target = row["作用靶点"]
            component = row["包含成分"]
            dose = row["起效剂量mg"]
            self.target_to_components.setdefault(target, []).append(
                (component, int(dose))
            )

    def _build_substance_mapping(self) -> None:
        """
        构建两套映射：
        1. 成分 → 物质列表: component_to_substances
        2. 物质 → 详细信息: substance_info

        映射结果示例:
            component_to_substances:
                {"Component_X": [("枸杞", 0.10, 0.80),
                                 ("桑叶", 0.15, 0.90)]}
            substance_info:
                {"枸杞": {"包含成分": "Component_X",
                          "成分含量占比": 0.10,
                          "消化吸收率": 0.80}}
        """
        for _, row in self.substance_df.iterrows():
            substance = row["物质名称"]
            component = row["包含成分"]
            ratio = float(row["成分含量占比"])
            absorption = float(row["消化吸收率"])

            # 成分 -> 物质
            self.component_to_substances.setdefault(component, []).append(
                (substance, ratio, absorption)
            )
            # 物质 -> 详情
            self.substance_info[substance] = {
                "包含成分": component,
                "成分含量占比": ratio,
                "消化吸收率": absorption,
            }

    def _build_synergy_mapping(self) -> None:
        """
        构建 (物质A, 物质B) → 协同系数 双向映射。

        协同系数对顺序不敏感（A+B 与 B+A 等效），
        因此同时存储正序与逆序键值。
        映射结果示例:
            {("枸杞", "黄芪"): 1.20, ("黄芪", "枸杞"): 1.20,
             ("桑叶", "葛根"): 1.15, ("葛根", "桑叶"): 1.15}
        """
        for _, row in self.synergy_df.iterrows():
            sub_a = row["物质A"]
            sub_b = row["物质B"]
            coef = float(row["协同系数"])
            # 双向存储，顺序无关
            self.synergy_map[(sub_a, sub_b)] = coef
            self.synergy_map[(sub_b, sub_a)] = coef

    # ===================== 3. 公开查询接口 =====================

    def get_targets_by_disease(self, disease_name: str) -> List[str]:
        """
        根据疾病名称查询其作用靶点列表。

        Args:
            disease_name: 疾病名称，如 "糖尿病"

        Returns:
            靶点列表，如 ["Target_A", "Target_B"]；若疾病不存在则返回空列表
        """
        return self.disease_to_targets.get(disease_name, [])

    def get_components_by_target(self, target: str) -> List[Tuple[str, int]]:
        """
        根据靶点查询对应成分及起效剂量。

        Args:
            target: 靶点名称，如 "Target_A"

        Returns:
            [(成分名, 起效剂量mg), ...] 列表
        """
        return self.target_to_components.get(target, [])

    def get_substances_by_component(
        self, component: str
    ) -> List[Tuple[str, float, float]]:
        """
        根据成分查询含该成分的物质列表。

        Args:
            component: 成分名称，如 "Component_X"

        Returns:
            [(物质名, 含量占比, 消化吸收率), ...] 列表
        """
        return self.component_to_substances.get(component, [])

    def get_substance_detail(self, substance: str) -> Optional[Dict]:
        """
        查询单个物质的完整信息。

        Args:
            substance: 物质名称，如 "枸杞"

        Returns:
            包含 包含成分、成分含量占比、消化吸收率 的字典；不存在则返回 None
        """
        return self.substance_info.get(substance)

    def get_synergy(self, sub_a: str, sub_b: str) -> float:
        """
        查询两种物质之间的协同系数。

        Args:
            sub_a: 物质A 名称
            sub_b: 物质B 名称

        Returns:
            协同系数，默认 1.0（无协同效应时按独立作用计算）
        """
        return self.synergy_map.get((sub_a, sub_b), 1.0)

    def resolve_disease_to_substances(
        self, disease_name: str
    ) -> List[Dict]:
        """
        一站式查询：疾病 → 靶点 → 成分 → 物质。

        沿映射链路递归展开，返回所有可作用于此疾病的物质及其属性。

        Args:
            disease_name: 疾病名称

        Returns:
            [{"物质": str, "成分": str, "靶点": str, "起效剂量mg": int,
              "含量占比": float, "吸收率": float}, ...]
        """
        results = []
        seen = set()  # 去重
        for target in self.get_targets_by_disease(disease_name):
            for comp, dose in self.get_components_by_target(target):
                for sub, ratio, absorb in self.get_substances_by_component(comp):
                    key = (sub, target)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "物质": sub,
                        "成分": comp,
                        "靶点": target,
                        "起效剂量mg": dose,
                        "含量占比": ratio,
                        "吸收率": absorb,
                    })
        return results

    # ===================== 4. 信息展示 =====================

    def summary(self) -> str:
        """打印数据概览，便于调试与确认加载正确。"""
        lines = [
            "=" * 50,
            "数据加载概览",
            "=" * 50,
            f"疾病种类: {len(self.disease_to_targets)}",
            f"靶点种类: {len(self.target_to_components)}",
            f"成分种类: {len(self.component_to_substances)}",
            f"物质种类: {len(self.substance_info)}",
            f"协同组合: {len(self.synergy_df)}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ===================== 模块自测 =====================
if __name__ == "__main__":
    # 不传参数，自动基于本文件位置推算项目根目录并定位 data/data.xlsx
    loader = DataLoader()
    print(loader.summary())

    # 示例查询
    print("\n[查询] 糖尿病 → 靶点:", loader.get_targets_by_disease("糖尿病"))
    print("[查询] Target_A → 成分:", loader.get_components_by_target("Target_A"))
    print("[查询] Component_X → 物质:", loader.get_substances_by_component("Component_X"))
    print("[查询] 枸杞 + 黄芪 协同系数:", loader.get_synergy("枸杞", "黄芪"))

    print("\n[一站式] 糖尿病可作用物质:")
    for item in loader.resolve_disease_to_substances("糖尿病"):
        print(f"  {item}")
