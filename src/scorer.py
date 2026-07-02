"""
药食同源饮品辅助配方推荐系统 —— 原料评分模块。

评分用于优化前的候选排序和解释展示；真正的剂量组合仍交给 optimizer.py 中
的 PuLP 模型求解。
"""

from typing import Dict, List, Optional

from src.data_loader import DataLoader
from src.matcher import DiseaseMatcher


class SubstanceScorer:
    """根据有效吸收、靶点覆盖和协同关系对候选原料评分。"""

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data = data_loader if data_loader is not None else DataLoader()
        self.matcher = DiseaseMatcher(self.data)

    def score(self, disease_name: str) -> List[Dict]:
        """
        对指定疾病的候选原料进行评分。

        评分由三部分组成：
        - 有效吸收能力：含量占比 * 消化吸收率
        - 靶点覆盖能力：同一原料覆盖的靶点数量
        - 协同潜力：候选集合内可形成协同组合的平均增益
        """
        candidates = self.matcher.match_substances(disease_name)
        if not candidates:
            return []

        substances = sorted({item["物质"] for item in candidates})
        target_count = {
            sub: len({item["靶点"] for item in candidates if item["物质"] == sub})
            for sub in substances
        }
        max_target_count = max(target_count.values()) if target_count else 1

        scored = []
        for sub in substances:
            rows = [item for item in candidates if item["物质"] == sub]
            best_absorption = max(item["每克有效吸收量mg"] for item in rows)
            absorption_score = best_absorption / 1000.0
            coverage_score = target_count[sub] / max_target_count
            synergy_score = self._synergy_potential(sub, substances)

            final_score = (
                0.50 * absorption_score
                + 0.35 * coverage_score
                + 0.15 * synergy_score
            )
            scored.append({
                "物质": sub,
                "评分": round(final_score, 4),
                "有效吸收评分": round(absorption_score, 4),
                "靶点覆盖评分": round(coverage_score, 4),
                "协同潜力评分": round(synergy_score, 4),
                "覆盖靶点": sorted({item["靶点"] for item in rows}),
                "关联成分": sorted({item["成分"] for item in rows}),
            })

        scored.sort(key=lambda item: item["评分"], reverse=True)
        return scored

    def _synergy_potential(self, substance: str, candidates: List[str]) -> float:
        """计算某原料在候选集合中的协同潜力，归一到 0-1 附近。"""
        gains = []
        for other in candidates:
            if other == substance:
                continue
            coef = self.data.get_synergy(substance, other)
            if coef > 1.0:
                gains.append(coef - 1.0)
        if not gains:
            return 0.0
        return min(sum(gains) / len(gains), 1.0)


if __name__ == "__main__":
    scorer = SubstanceScorer()
    print(scorer.score("糖尿病"))
