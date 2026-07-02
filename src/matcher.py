"""
药食同源饮品辅助配方推荐系统 —— 疾病匹配模块。

负责把用户选择的疾病展开为：
疾病 -> 靶点 -> 成分 -> 候选原料
该模块只做匹配与结构化整理，不负责评分和优化。
"""

from typing import Dict, List, Optional

from src.data_loader import DataLoader


class DiseaseMatcher:
    """疾病到靶点、成分、原料的匹配器。"""

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data = data_loader if data_loader is not None else DataLoader()

    def match_targets(self, disease_name: str) -> List[str]:
        """根据疾病名称获取关联靶点。"""
        return self.data.get_targets_by_disease(disease_name)

    def match_components(self, disease_name: str) -> List[Dict]:
        """根据疾病名称获取关联成分及其靶点剂量信息。"""
        components = []
        seen = set()
        for target in self.match_targets(disease_name):
            for component, dose in self.data.get_components_by_target(target):
                key = (target, component)
                if key in seen:
                    continue
                seen.add(key)
                components.append({
                    "疾病": disease_name,
                    "靶点": target,
                    "成分": component,
                    "起效剂量mg": dose,
                })
        return components

    def match_substances(self, disease_name: str) -> List[Dict]:
        """
        获取疾病对应的所有候选原料。

        Returns:
            [{"疾病": str, "靶点": str, "成分": str, "物质": str,
              "起效剂量mg": int, "含量占比": float, "吸收率": float,
              "每克有效吸收量mg": float}, ...]
        """
        candidates = []
        for item in self.data.resolve_disease_to_substances(disease_name):
            per_gram = 1000.0 * float(item["含量占比"]) * float(item["吸收率"])
            candidates.append({
                "疾病": disease_name,
                "靶点": item["靶点"],
                "成分": item["成分"],
                "物质": item["物质"],
                "起效剂量mg": item["起效剂量mg"],
                "含量占比": item["含量占比"],
                "吸收率": item["吸收率"],
                "每克有效吸收量mg": per_gram,
            })
        return candidates

    def get_match_report(self, disease_name: str) -> Dict:
        """生成疾病匹配报告，供 Streamlit 展示或调试使用。"""
        targets = self.match_targets(disease_name)
        components = self.match_components(disease_name)
        substances = self.match_substances(disease_name)
        return {
            "疾病": disease_name,
            "靶点": targets,
            "成分": components,
            "候选原料": substances,
            "统计": {
                "靶点数": len(set(targets)),
                "成分数": len({item["成分"] for item in components}),
                "候选原料数": len({item["物质"] for item in substances}),
            },
        }


if __name__ == "__main__":
    matcher = DiseaseMatcher()
    print(matcher.get_match_report("糖尿病"))
