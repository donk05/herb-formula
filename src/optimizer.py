"""
药食同源饮品辅助配方推荐系统 —— 配方优化模块
基于 pulp 构建混合整数线性规划（MILP）模型，求解最小重量配方。
"""

from typing import Dict, List, Tuple, Optional
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpStatus,
    PULP_CBC_CMD,
)

# 动态推算项目根目录，以便在任何工作目录下均可导入 data_loader
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data_loader import DataLoader


class FormulaOptimizer:
    """
    配方优化器：针对指定疾病和剂型，求解满足所有靶点起效剂量的最小总重量配方。

    数学模型:
        min  Σ x_i                              （总重量最小化）
        s.t.  Σ (x_i * p_i + synergy_bonus_i) >= dose_t   ∀ 靶点 t
              Σ x_i <= 5                         （仅代茶饮，模拟茶包容量上限）
              x_i >= 0

        其中:
          x_i     = 物质 i 的添加重量 (g)
          p_i     = 1000 * 含量占比 * 消化吸收率 (mg/g, 每克物质的有效吸收量)
          synergy_bonus_i = 协同奖励项，仅当协同对双方均被选用时激活
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Args:
            data_loader: DataLoader 实例；若为 None 则自动创建
        """
        self.data = data_loader if data_loader is not None else DataLoader()

    # ===================== 主求解入口 =====================

    def optimize(
        self, disease_name: str, formula_type: str = "代茶饮",
        min_dose: float = 0.5
    ) -> Dict:
        """
        求解最优配方。

        Args:
            disease_name: 疾病名称，如 "糖尿病"
            formula_type: 剂型，"代茶饮"（总重 ≦ 5g，模拟独立茶包）或
                          "汤剂"（无总重限制，模拟传统煎煮）
            min_dose: 单物质最低用量 (g)，防止模型以极微量骗取协同奖励。
                      默认 0.5g，设 0 则不限制。

        Returns:
            {
                "状态": str,           # Optimal / Infeasible / ...
                "总重量_g": float,
                "配方": [{"物质": str, "重量_g": float, "有效吸收量_mg": float}, ...],
                "靶点满足情况": [{"靶点": str, "需要_mg": int, "实际_mg": float}, ...],
            }
        """
        # 1. 获取疾病关联的所有候选记录
        resolved = self.data.resolve_disease_to_substances(disease_name)
        if not resolved:
            return {"状态": "无候选物质", "总重量_g": 0, "配方": [], "靶点满足情况": []}

        # 2. 提取候选物质集合 & 按靶点聚合
        substances = list({r["物质"] for r in resolved})
        # target_info: {靶点: {"剂量": dose, "物质": [(sub, per_gram), ...]}}
        target_info: Dict[str, Dict] = {}
        for r in resolved:
            t = r["靶点"]
            sub = r["物质"]
            dose = int(r["起效剂量mg"])
            per_gram = 1000.0 * float(r["含量占比"]) * float(r["吸收率"])  # mg/g
            if t not in target_info:
                target_info[t] = {"剂量": dose, "物质": []}
            # 同一靶点可能来自多条 Mechanism 记录，取最大剂量
            target_info[t]["剂量"] = max(target_info[t]["剂量"], dose)
            target_info[t]["物质"].append((sub, per_gram))

        # 去重：同一靶点下同一物质只保留一条（per_gram 相同，取首次）
        for t in target_info:
            seen = set()
            dedup = []
            for sub, pg in target_info[t]["物质"]:
                if sub not in seen:
                    seen.add(sub)
                    dedup.append((sub, pg))
            target_info[t]["物质"] = dedup

        # 3. 获取仅涉及候选物质的协同对（去重，按字母序保证 a < b）
        synergy_pairs: List[Tuple[str, str, float]] = []
        seen_pairs = set()
        for (a, b), coef in self.data.synergy_map.items():
            if a in substances and b in substances:
                key = (a, b) if a < b else (b, a)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    synergy_pairs.append((key[0], key[1], coef))

        # 4. 构建 MILP 模型
        prob = LpProblem("FormulaOptimization", LpMinimize)

        # ---- 4a. 决策变量 ----
        x = {s: LpVariable(f"weight_{s}", lowBound=0, cat="Continuous")
             for s in substances}
        z = {s: LpVariable(f"use_{s}", cat="Binary")
             for s in substances}

        M_weight = 5.0 if formula_type == "代茶饮" else 500.0
        for s in substances:
            prob += x[s] <= M_weight * z[s], f"link_xz_upper_{s}"
            if min_dose > 0:
                # 若选用某物质，其用量至少达到最低实用剂量，防止微量骗取协同
                prob += x[s] >= min_dose * z[s], f"link_xz_lower_{s}"

        # ---- 4b. 协同变量（big-M 线性化） ----
        M_bonus = 10000.0  # 足够大的常数
        y = {}      # y[idx]: 协同对是否激活
        bonus = {}  # bonus[(substance, idx)]: 协同额外贡献

        for idx, (a, b, coef) in enumerate(synergy_pairs):
            y[idx] = LpVariable(f"synergy_active_{idx}", cat="Binary")

            # 仅当双方均被选用时协同才可能激活
            prob += y[idx] <= z[a], f"syn_{idx}_link_{a}"
            prob += y[idx] <= z[b], f"syn_{idx}_link_{b}"
            prob += y[idx] >= z[a] + z[b] - 1, f"syn_{idx}_both_{a}_{b}"

            # 物质 A 的协同奖励
            pg_a = self._per_gram(a)
            bonus_a = LpVariable(f"bonus_{a}_syn{idx}", lowBound=0, cat="Continuous")
            prob += bonus_a <= pg_a * (coef - 1) * x[a], f"bonus_{a}_natural_{idx}"
            prob += bonus_a <= M_bonus * y[idx], f"bonus_{a}_bigM_{idx}"
            bonus[(a, idx)] = bonus_a

            # 物质 B 的协同奖励
            pg_b = self._per_gram(b)
            bonus_b = LpVariable(f"bonus_{b}_syn{idx}", lowBound=0, cat="Continuous")
            prob += bonus_b <= pg_b * (coef - 1) * x[b], f"bonus_{b}_natural_{idx}"
            prob += bonus_b <= M_bonus * y[idx], f"bonus_{b}_bigM_{idx}"
            bonus[(b, idx)] = bonus_b

        # ---- 4c. 靶点约束：每个靶点的有效吸收量 >= 起效剂量 ----
        target_results = {}  # 记录求解后验证用
        for t, info in target_info.items():
            dose = info["剂量"]
            expr = 0
            for sub, pg in info["物质"]:
                # 基础贡献
                expr += x[sub] * pg
                # 协同奖励（该物质所属的所有协同对）
                for idx, (a, b, _) in enumerate(synergy_pairs):
                    if sub == a or sub == b:
                        expr += bonus.get((sub, idx), 0)
            prob += expr >= dose, f"target_{t}"
            target_results[t] = {"剂量": dose, "表达式": expr}

        # ---- 4d. 剂型约束 ----
        if formula_type == "代茶饮":
            prob += lpSum([x[s] for s in substances]) <= 5.0, "tea_bag_limit"

        # ---- 4e. 目标函数 ----
        prob += lpSum([x[s] for s in substances]), "total_weight"

        # 5. 求解
        prob.solve(PULP_CBC_CMD(msg=False))

        # 6. 整理结果
        status_str = LpStatus[prob.status]
        total_weight = lpSum([x[s] for s in substances]).value()

        formula = []
        for s in substances:
            w = x[s].value()
            if w and w > min_dose / 2:  # 过滤未达最低用量的微量残留
                # 计算该物质的总有效吸收量（基础 + 协同奖励）
                pg = self._per_gram(s)
                base_abs = w * pg
                syn_bonus_total = 0.0
                for idx, (a, b, _) in enumerate(synergy_pairs):
                    if s == a or s == b:
                        b_val = bonus.get((s, idx))
                        if b_val is not None:
                            syn_bonus_total += b_val.value() or 0.0
                formula.append({
                    "物质": s,
                    "重量_g": round(w, 4),
                    "有效吸收量_mg": round(base_abs + syn_bonus_total, 2),
                    "其中协同奖励_mg": round(syn_bonus_total, 2),
                })

        # 靶点满足情况
        target_satisfaction = []
        for t, info in target_info.items():
            actual = sum(
                item["有效吸收量_mg"]
                for item in formula
                if any(item["物质"] == sub for sub, _ in info["物质"])
            )
            target_satisfaction.append({
                "靶点": t,
                "需要_mg": info["剂量"],
                "实际_mg": round(actual, 2),
                "满足": actual >= info["剂量"] - 1e-6,
            })

        return {
            "状态": status_str,
            "总重量_g": round(total_weight, 4),
            "配方": formula,
            "靶点满足情况": target_satisfaction,
        }

    # ===================== 辅助方法 =====================

    def _per_gram(self, substance: str) -> float:
        """计算某物质每克提供的有效吸收量 (mg/g)。"""
        info = self.data.get_substance_detail(substance)
        if info is None:
            return 0.0
        return 1000.0 * info["成分含量占比"] * info["消化吸收率"]

    # ===================== 结果展示 =====================

    def print_result(self, result: Dict) -> None:
        """格式化打印优化结果。"""
        print("\n" + "=" * 50)
        print(f"求解状态: {result['状态']}")
        print(f"配方总重量: {result['总重量_g']} g")
        print("-" * 50)
        print(f"{'物质':<8} {'重量(g)':<12} {'有效吸收(mg)':<16} {'协同奖励(mg)':<14}")
        print("-" * 50)
        for item in result["配方"]:
            print(
                f"{item['物质']:<8} "
                f"{item['重量_g']:<12.4f} "
                f"{item['有效吸收量_mg']:<16.2f} "
                f"{item['其中协同奖励_mg']:<14.2f}"
            )
        print("-" * 50)
        print("靶点满足情况:")
        for t in result["靶点满足情况"]:
            mark = "OK" if t["满足"] else "!! 未满足"
            print(f"  {t['靶点']}: 需要 {t['需要_mg']} mg, "
                  f"实际 {t['实际_mg']} mg  [{mark}]")
        print("=" * 50)


# ===================== 模块自测 =====================
if __name__ == "__main__":
    optimizer = FormulaOptimizer()

    # 测试 1：糖尿病 + 代茶饮
    print("\n" + "█" * 50)
    print("  测试：糖尿病 × 代茶饮")
    print("█" * 50)
    result1 = optimizer.optimize("糖尿病", "代茶饮")
    optimizer.print_result(result1)

    # 测试 2：糖尿病 + 汤剂（无总量限制）
    print("\n" + "█" * 50)
    print("  测试：糖尿病 × 汤剂")
    print("█" * 50)
    result2 = optimizer.optimize("糖尿病", "汤剂")
    optimizer.print_result(result2)

    # 测试 3：高血压 + 代茶饮
    print("\n" + "█" * 50)
    print("  测试：高血压 × 代茶饮")
    print("█" * 50)
    result3 = optimizer.optimize("高血压", "代茶饮")
    optimizer.print_result(result3)
