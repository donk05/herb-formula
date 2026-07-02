"""
药食同源饮品辅助配方推荐系统 —— 剂量与约束检查模块。

用于对 optimizer.py 输出的配方结果做二次校验，便于前端提示和论文/答辩中
解释模型是否满足剂型、最低用量、靶点剂量等约束。
"""

from typing import Dict, List, Optional

from src.data_loader import DataLoader


class ConstraintChecker:
    """检查配方优化结果是否满足业务约束。"""

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data = data_loader if data_loader is not None else DataLoader()

    def check(
        self,
        result: Dict,
        formula_type: str = "代茶饮",
        min_dose: float = 0.5,
    ) -> Dict:
        """
        检查优化结果。

        Args:
            result: FormulaOptimizer.optimize 返回的结果字典。
            formula_type: 剂型，代茶饮总重量上限为 5g。
            min_dose: 单个被选原料的最低用量。

        Returns:
            {"通过": bool, "问题": [...], "明细": {...}}
        """
        issues: List[str] = []
        formula = result.get("配方", [])
        total_weight = float(result.get("总重量_g", 0) or 0)

        if result.get("状态") != "Optimal":
            issues.append(f"求解状态不是 Optimal：{result.get('状态')}")

        if formula_type == "代茶饮" and total_weight > 5.0 + 1e-6:
            issues.append(f"代茶饮总重量超过 5g：{total_weight:.4f}g")

        for item in formula:
            weight = float(item.get("重量_g", 0) or 0)
            substance = item.get("物质", "")
            if weight < min_dose - 1e-6:
                issues.append(f"{substance} 用量低于最低用量 {min_dose}g：{weight:.4f}g")

        for target in result.get("靶点满足情况", []):
            if not target.get("满足", False):
                issues.append(
                    f"{target.get('靶点')} 未满足起效剂量："
                    f"需要 {target.get('需要_mg')}mg，实际 {target.get('实际_mg')}mg"
                )

        return {
            "通过": len(issues) == 0,
            "问题": issues,
            "明细": {
                "剂型": formula_type,
                "总重量_g": total_weight,
                "最低用量_g": min_dose,
                "原料数": len(formula),
            },
        }


if __name__ == "__main__":
    checker = ConstraintChecker()
    demo = {"状态": "Optimal", "总重量_g": 0, "配方": [], "靶点满足情况": []}
    print(checker.check(demo))
