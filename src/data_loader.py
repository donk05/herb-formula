"""
药食同源饮品配方推荐系统 —— 知识图谱数据加载模块
读取「药食同源_真实关系链」Excel，构建 中药→化合物→靶点→疾病 多级映射。
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 动态推算项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 中文疾病名 → 新表中的英文疾病名关键词（精确匹配优先）
CN_TO_EN_DISEASE = {
    "糖尿病": "Diabetes Mellitus",
    "高血压": "Essential Hypertension",
    "高血脂": "Hyperlipidemia",
    "肥胖症": "Obesity",
    "失眠": "Sleep Deprivation",
    "消化不良": "Dyspepsia",
    "免疫力低下": "Immune System Diseases",
    "肝损伤": "Liver diseases",
    "痛风": "Gout",
    "慢性咽炎": "Pharyngitis",
    "慢性胃炎": "Gastritis, Atrophic",
    "贫血": "Anemia",
    "痴呆": "Dementia",
    "阿尔茨海默": "Alzheimer's Disease",
    "冠心病": "Coronary Artery Disease",
    "抑郁症": "Depression",
    "哮喘": "Asthma",
    "类风湿": "Rheumatoid Arthritis",
    "帕金森": "Parkinson Disease",
    "胃炎": "Gastritis",
    "肝炎": "Hepatitis",
    "肾炎": "Nephritis",
    "肺炎": "Pneumonia",
    "中风": "Stroke",
    "便秘": "Constipation",
    "腹泻": "Diarrhea",
    "咳嗽": "Cough",
    "感冒": "Common Cold",
    "过敏": "Hypersensitivity",
    "衰老": "Aging",
    "疲劳": "Fatigue",
    "脱发": "Alopecia",
    "痤疮": "Acne Vulgaris",
    "银屑病": "Psoriasis",
    "白癜风": "Vitiligo",
    "骨质疏松": "Osteoporosis",
    "更年期": "Menopause",
    "前列腺": "Prostatic Hyperplasia",
    "乳腺癌": "Breast Cancer",
    "肺癌": "Lung Cancer",
    "胃癌": "Stomach Cancer",
    "肝癌": "Liver Cancer",
    "结肠癌": "Colon Cancer",
    "胰腺癌": "Pancreatic Cancer",
    "食管癌": "Esophageal Cancer",
    "白血病": "Leukemia",
    "淋巴瘤": "Lymphoma",
    "黑色素瘤": "Melanoma",
    "甲状腺": "Thyroid",
    "胰腺炎": "Pancreatitis",
    "支气管炎": "Bronchitis",
    "鼻窦炎": "Sinusitis",
    "关节炎": "Arthritis",
    "脊柱炎": "Spondylitis",
    "心肌梗死": "Myocardial Infarction",
    "心力衰竭": "Heart Failure",
    "心律失常": "Arrhythmia",
    "动脉硬化": "Atherosclerosis",
    "静脉血栓": "Venous Thrombosis",
    "中风": "Stroke",
    "偏头痛": "Migraine",
    "癫痫": "Epilepsy",
    "多发性硬化": "Multiple Sclerosis",
    "肌萎缩": "Muscular Atrophy",
    "红斑狼疮": "Lupus Erythematosus",
    "硬皮病": "Scleroderma",
    "克罗恩": "Crohn Disease",
    "溃疡性结肠炎": "Ulcerative Colitis",
    "肠易激": "Irritable Bowel Syndrome",
    "肾衰竭": "Renal Failure",
    "肾结石": "Kidney Calculi",
    "尿路感染": "Urinary Tract Infection",
    "前列腺癌": "Prostatic Cancer",
    "子宫肌瘤": "Uterine Fibroids",
    "子宫内膜异位": "Endometriosis",
    "多囊卵巢": "Polycystic Ovary Syndrome",
    "不孕": "Infertility",
    "青光眼": "Glaucoma",
    "白内障": "Cataract",
    "黄斑变性": "Macular Degeneration",
    "视网膜": "Retinal",
    "耳聋": "Deafness",
    "耳鸣": "Tinnitus",
    "龋齿": "Dental Caries",
    "牙周炎": "Periodontitis",
    "湿疹": "Eczema",
    "荨麻疹": "Urticaria",
    "带状疱疹": "Herpes Zoster",
    "真菌感染": "Fungal Infection",
    "疟疾": "Malaria",
    "结核": "Tuberculosis",
    "艾滋病": "HIV",
    "乙肝": "Hepatitis B",
    "丙肝": "Hepatitis C",
    "甲亢": "Hyperthyroidism",
    "甲减": "Hypothyroidism",
    "库欣": "Cushing Syndrome",
    "生长障碍": "Growth Disorders",
    "发育迟缓": "Developmental Delay",
    "自闭症": "Autism",
    "多动症": "Attention Deficit",
    "精神分裂": "Schizophrenia",
    "双相": "Bipolar Disorder",
    "焦虑症": "Anxiety Disorder",
    "强迫症": "Obsessive-Compulsive",
    "创伤后应激": "Post-Traumatic Stress",
    "厌食症": "Anorexia",
    "贪食症": "Bulimia",
    "酒精依赖": "Alcoholism",
    "药物成瘾": "Drug Addiction",
    "败血症": "Sepsis",
    "休克": "Shock",
    "烧伤": "Burns",
    "创伤": "Wounds and Injuries",
    "骨折": "Fractures",
    "椎间盘": "Intervertebral Disc",
    "坐骨神经": "Sciatica",
    "腕管": "Carpal Tunnel Syndrome",
    "坏死": "Necrosis",
    "纤维化": "Fibrosis",
    "水肿": "Edema",
    "出血": "Hemorrhage",
    "血栓": "Thrombosis",
    "栓塞": "Embolism",
    "缺血": "Ischemia",
    "梗死": "Infarction",
}

# 中文医学术语 → 英文搜索关键词（用于中文搜索回退）
CN_MEDICAL_TERMS = {
    "生长": "Growth",
    "障碍": "Disorder",
    "癌": "Cancer",
    "瘤": "Tumor",
    "炎": "Inflammation",
    "病": "Disease",
    "综合征": "Syndrome",
    "综合症": "Syndrome",
    "衰竭": "Failure",
    "硬化": "Sclerosis",
    "坏死": "Necrosis",
    "变性": "Degeneration",
    "萎缩": "Atrophy",
    "肥大": "Hypertrophy",
    "畸形": "Malformation",
    "缺陷": "Deficiency",
    "缺乏": "Deficiency",
    "感染": "Infection",
    "中毒": "Poisoning",
    "损伤": "Injury",
    "创伤": "Trauma",
    "出血": "Hemorrhage",
    "血栓": "Thrombosis",
    "梗死": "Infarction",
    "缺血": "Ischemia",
    "水肿": "Edema",
    "发热": "Fever",
    "疼痛": "Pain",
    "痉挛": "Spasm",
    "麻痹": "Paralysis",
    "昏迷": "Coma",
    "眩晕": "Vertigo",
    "头痛": "Headache",
    "腹痛": "Abdominal Pain",
    "胸痛": "Chest Pain",
    "背痛": "Back Pain",
    "关节": "Joint",
    "肌肉": "Muscle",
    "骨骼": "Bone",
    "皮肤": "Skin",
    "心脏": "Heart",
    "血管": "Vascular",
    "肺": "Lung",
    "肝": "Liver",
    "肾": "Kidney",
    "胃": "Stomach",
    "肠": "Intestinal",
    "脑": "Brain",
    "神经": "Neural",
    "内分泌": "Endocrine",
    "免疫": "Immune",
    "血液": "Blood",
    "代谢": "Metabolic",
    "遗传": "Hereditary",
    "先天": "Congenital",
    "慢性": "Chronic",
    "急性": "Acute",
    "原发性": "Primary",
    "继发性": "Secondary",
    "良性": "Benign",
    "恶性": "Malignant",
    "转移": "Metastatic",
    "复发": "Recurrent",
    "难治": "Refractory",
    "儿童": "Childhood",
    "青少年": "Adolescent",
    "老年": "Senile",
    "妊娠": "Pregnancy",
    "产后": "Postpartum",
    "肥胖": "Obesity",
    "消瘦": "Wasting",
    "成瘾": "Addiction",
    "依赖": "Dependence",
    "过敏": "Allergic",
    "自身免疫": "Autoimmune",
    "发育": "Development",
    "迟缓": "Delay",
    "生长": "Growth",
    "障碍": "Disorder",
    "皮肤": "Skin",
    "炎症": "Inflammatory",
    "儿童": "Child",
    "痴呆": "Dementia",
    "记忆": "Memory",
    "认知": "Cognitive",
    "抑郁": "Depressive",
    "焦虑": "Anxiety",
    "睡眠": "Sleep",
    "疲劳": "Fatigue",
    "腹泻": "Diarrhea",
    "便秘": "Constipation",
    "呕吐": "Vomiting",
    "恶心": "Nausea",
    "食欲": "Appetite",
    "体重": "Weight",
    "血压": "Blood Pressure",
    "血糖": "Blood Glucose",
    "血脂": "Lipid",
    "尿酸": "Uric Acid",
    "胆固醇": "Cholesterol",
    "视力": "Vision",
    "听力": "Hearing",
    "言语": "Speech",
    "运动": "Motor",
    "感觉": "Sensory",
    "呼吸": "Respiratory",
    "消化": "Digestive",
    "泌尿": "Urinary",
    "生殖": "Reproductive",
    "月经": "Menstrual",
    "性功能": "Sexual",
    "脱发": "Hair Loss",
    "痤疮": "Acne",
    "皮疹": "Rash",
    "瘙痒": "Pruritus",
    "干燥": "Dry",
    "出汗": "Sweating",
    "结节": "Nodule",
    "囊肿": "Cyst",
    "息肉": "Polyp",
    "溃疡": "Ulcer",
    "糜烂": "Erosion",
    "增生": "Hyperplasia",
    "化生": "Metaplasia",
    "异型": "Dysplasia",
    "原位": "in situ",
    "浸润": "Invasive",
    "分化": "Differentiated",
    "未分化": "Undifferentiated",
}

# ==================== 英文→中文 词级翻译词典 ====================
# 从独立模块导入专业医学词典
from src.medical_dict import EN_TO_CN_MEDICAL

# 合并所有词源：CN_MEDICAL_TERMS 反向 + 专业医学词典
EN_TO_CN_WORD = {}
for cn, en in CN_MEDICAL_TERMS.items():
    for word in en.split():
        EN_TO_CN_WORD[word.lower()] = cn
    EN_TO_CN_WORD[en.lower()] = cn
EN_TO_CN_WORD.update({k.lower(): v for k, v in EN_TO_CN_MEDICAL.items()})

# CN_TO_EN_DISEASE 反向精确映射
EN_TO_CN_DISEASE_EXACT = {v: k for k, v in CN_TO_EN_DISEASE.items()}

# 补充额外英文词 → 中文词
# _EXTRA_EN_CN 已迁移至 src/medical_dict.py 的 EN_TO_CN_MEDICAL 中
_EXTRA_EN_CN_DEPRECATED = {
    "disease": "病", "diseases": "病", "disorder": "障碍", "disorders": "障碍",
    "syndrome": "综合征", "deficiency": "缺乏症", "deficient": "缺乏",
    "type": "型", "types": "型", "acute": "急性", "chronic": "慢性",
    "primary": "原发性", "secondary": "继发性", "benign": "良性",
    "malignant": "恶性", "metastatic": "转移性", "recurrent": "复发性",
    "refractory": "难治性", "congenital": "先天性", "hereditary": "遗传性",
    "familial": "家族性", "sporadic": "散发性", "idiopathic": "特发性",
    "juvenile": "青少年", "infantile": "婴儿", "neonatal": "新生儿",
    "senile": "老年性", "adult": "成人", "childhood": "儿童期",
    "pregnancy": "妊娠", "postpartum": "产后", "menstrual": "月经",
    "cancer": "癌", "carcinoma": "癌", "tumor": "瘤", "neoplasm": "肿瘤",
    "leukemia": "白血病", "lymphoma": "淋巴瘤", "sarcoma": "肉瘤",
    "melanoma": "黑色素瘤", "adenoma": "腺瘤", "adenocarcinoma": "腺癌",
    "infection": "感染", "inflammation": "炎", "inflammatory": "炎性",
    "hepatitis": "肝炎", "nephritis": "肾炎", "gastritis": "胃炎",
    "pancreatitis": "胰腺炎", "bronchitis": "支气管炎", "sinusitis": "鼻窦炎",
    "arthritis": "关节炎", "dermatitis": "皮炎", "myositis": "肌炎",
    "colitis": "结肠炎", "esophagitis": "食管炎", "meningitis": "脑膜炎",
    "encephalitis": "脑炎", "myocarditis": "心肌炎", "endocarditis": "心内膜炎",
    "pneumonia": "肺炎", "tuberculosis": "结核", "asthma": "哮喘",
    "diabetes": "糖尿病", "mellitus": "糖尿病", "hypertension": "高血压",
    "hyperlipidemia": "高血脂", "obesity": "肥胖", "anemia": "贫血",
    "gout": "痛风", "stroke": "中风", "epilepsy": "癫痫", "migraine": "偏头痛",
    "depression": "抑郁症", "anxiety": "焦虑症", "schizophrenia": "精神分裂症",
    "dementia": "痴呆", "alzheimer": "阿尔茨海默", "parkinson": "帕金森",
    "insomnia": "失眠", "narcolepsy": "发作性睡病", "autism": "自闭症",
    "sclerosis": "硬化症", "atrophy": "萎缩", "hypertrophy": "肥大",
    "hyperplasia": "增生", "metaplasia": "化生", "dysplasia": "异型增生",
    "necrosis": "坏死", "apoptosis": "凋亡", "fibrosis": "纤维化",
    "cirrhosis": "肝硬化", "steatosis": "脂肪变性", "calcification": "钙化",
    "hemorrhage": "出血", "thrombosis": "血栓", "embolism": "栓塞",
    "infarction": "梗死", "ischemia": "缺血", "edema": "水肿",
    "failure": "衰竭", "insufficiency": "功能不全", "dysfunction": "功能障碍",
    "pain": "疼痛", "headache": "头痛", "migraine": "偏头痛",
    "vertigo": "眩晕", "tinnitus": "耳鸣", "deafness": "耳聋", "blindness": "失明",
    "fever": "发热", "fatigue": "疲劳", "weakness": "乏力", "cachexia": "恶病质",
    "heart": "心脏", "cardiac": "心脏", "coronary": "冠状动脉",
    "myocardial": "心肌", "vascular": "血管", "artery": "动脉", "venous": "静脉",
    "liver": "肝", "hepatic": "肝", "kidney": "肾", "renal": "肾",
    "lung": "肺", "pulmonary": "肺", "respiratory": "呼吸",
    "stomach": "胃", "gastric": "胃", "intestinal": "肠", "colon": "结肠",
    "brain": "脑", "cerebral": "脑", "neural": "神经", "neurologic": "神经",
    "spinal": "脊髓", "ocular": "眼", "retinal": "视网膜", "corneal": "角膜",
    "bone": "骨", "skeletal": "骨骼", "joint": "关节", "articular": "关节",
    "muscle": "肌肉", "muscular": "肌肉", "skin": "皮肤", "cutaneous": "皮肤",
    "thyroid": "甲状腺", "adrenal": "肾上腺", "pituitary": "垂体",
    "ovarian": "卵巢", "testicular": "睾丸", "prostatic": "前列腺",
    "breast": "乳腺", "uterine": "子宫", "cervical": "宫颈", "endometrial": "子宫内膜",
    "pancreatic": "胰腺", "esophageal": "食管", "colorectal": "结直肠",
    "bladder": "膀胱", "prostate": "前列腺", "ovary": "卵巢",
    "blood": "血液", "hematologic": "血液", "immune": "免疫",
    "metabolic": "代谢", "endocrine": "内分泌", "nutritional": "营养",
    "genetic": "遗传", "chromosomal": "染色体", "mitochondrial": "线粒体",
    "viral": "病毒性", "bacterial": "细菌性", "fungal": "真菌性",
    "parasitic": "寄生虫性", "autoimmune": "自身免疫", "allergic": "过敏性",
    "toxic": "中毒性", "traumatic": "创伤性", "iatrogenic": "医源性",
    "experimental": "实验性", "chemical": "化学性", "drug": "药物性",
    "alcohol": "酒精性", "smoking": "吸烟相关", "radiation": "放射性",
    "postoperative": "术后", "preoperative": "术前", "intraoperative": "术中",
    "complication": "并发症", "sequelae": "后遗症",
    "resistant": "耐药", "sensitive": "敏感", "intolerant": "不耐受",
    "advanced": "晚期", "early": "早期", "late": "晚期", "onset": "发病",
    "mild": "轻度", "moderate": "中度", "severe": "重度", "critical": "危重",
    "localized": "局限性", "generalized": "全身性", "systemic": "系统性",
    "unilateral": "单侧", "bilateral": "双侧", "multiple": "多发性",
    "isolated": "孤立性", "combined": "联合性", "mixed": "混合性",
    "transient": "一过性", "persistent": "持续性", "permanent": "永久性",
    "progressive": "进行性", "regressive": "退行性", "stable": "稳定性",
    "remission": "缓解期", "relapse": "复发期", "exacerbation": "加重期",
    "undifferentiated": "未分化", "differentiated": "分化",
    "anaplastic": "间变性", "pleomorphic": "多形性",
    "syndrome,": "综合征", "disease,": "病", "disorder,": "障碍",
}
# _EXTRA_EN_CN 已由 src/medical_dict.py 的 EN_TO_CN_MEDICAL 替代
# 下面函数已关联至新词典 (见第 ~283 行的 import 和 merge)


def translate_en_disease_to_cn(en_name: str) -> str:
    """将英文疾病名翻译为中文（精确匹配 + 智能词级翻译）。"""
    # 1. 精确匹配
    if en_name in EN_TO_CN_DISEASE_EXACT:
        return EN_TO_CN_DISEASE_EXACT[en_name]
    for en, cn in EN_TO_CN_DISEASE_EXACT.items():
        if en.lower() == en_name.lower():
            return cn

    import re

    # 2. 分词 + 智能翻译
    SKIP = {'of', 'the', 'a', 'an', 'and', 'or', 'not', 'with', 'without',
            'to', 'in', 'on', 'at', 'by', 'for', 'from', 'as', 'is', 'was',
            'were', 'are', 'be', 'due', 'per', 'no', 'associated', 'related',
            'induced', 'linked', 'caused', 'mediated', 'type', 'types'}
    ROMAN = {'i': 'I', 'ii': 'II', 'iii': 'III', 'iv': 'IV', 'v': 'V',
             'vi': 'VI', 'vii': 'VII', 'viii': 'VIII', 'ix': 'IX', 'x': 'X'}

    words = re.split(r'[\s,;()]+', en_name)
    cn_parts = []

    for w in words:
        if not w:
            continue
        wl = w.lower().strip("'\"-./")
        if not wl or wl in SKIP:
            continue
        # Roman 数字 → 中文
        if wl in ROMAN:
            cn_parts.append(ROMAN[wl] + "型")
            continue
        # 基因位点 (如 11q23, 17p13.3) → 跳过
        if re.match(r'^[0-9]+[pq][0-9]+(\.[0-9]+)?$', wl):
            continue
        # 酶/蛋白名 → 保留原文
        if re.match(r'.*(ase|ogen|ylase|tase|nase|dase|rin|mab|nib|mib|stat)[0-9]*$', wl, re.I):
            cn_parts.append(wl)
            continue
        # 查词典翻译
        t = EN_TO_CN_WORD.get(wl)
        if t is None:
            for sfx in ['s', 'es', 'ing', 'ed', 'ic', 'al', 'ive', 'ous', 'ary',
                        'sis', 'ia', 'ism', 'itis', 'oma', 'osis', 'pathy']:
                if wl.endswith(sfx) and len(wl) - len(sfx) > 2:
                    t = EN_TO_CN_WORD.get(wl[:-len(sfx)])
                    if t:
                        break
        if t is None:
            if '-' in wl and len(wl) > 4:
                subs = wl.split('-')
                trans_subs = []
                ok = True
                for sub in subs:
                    st = EN_TO_CN_WORD.get(sub)
                    if st:
                        trans_subs.append(st)
                    elif re.match(r'^[0-9]+$', sub):
                        trans_subs.append(sub)
                    else:
                        ok = False
                        break
                t = ''.join(trans_subs) if ok else wl
            else:
                t = wl
        cn_parts.append(t)

    result = "".join(cn_parts)
    if result and sum(1 for c in result if '一' <= c <= '鿿') / len(result) < 0.3:
        return en_name
    return result


class GraphDataLoader:
    """
    知识图谱数据加载器。

    关系链: 中药 → 化合物 → 靶点 → 疾病
    支持中文疾病名查询，自动映射到英文后在图谱中检索。
    """

    def __init__(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(
                _PROJECT_ROOT, "data", "药食同源_真实关系链（去重）(1).xlsx"
            )
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.filepath}")

        # 原始 DataFrame
        self.herb_compound_df: pd.DataFrame = None
        self.compound_target_df: pd.DataFrame = None
        self.target_disease_df: pd.DataFrame = None

        # 核心映射字典
        self.disease_to_targets: Dict[str, Set[str]] = {}     # 疾病 → 靶点ID集合
        self.target_to_compounds: Dict[str, Set[str]] = {}    # 靶点ID → 化合物ID集合
        self.compound_to_herbs: Dict[str, Set[str]] = {}      # 化合物ID → 中药名集合
        self.herb_to_compounds: Dict[str, Set[str]] = {}      # 中药 → 化合物ID集合

        # 元数据
        self.target_names: Dict[str, str] = {}   # 靶点ID → 靶点名称
        self.compound_names: Dict[str, str] = {} # 化合物ID → 化合物名称
        self.disease_names: Dict[str, str] = {}  # 疾病ID → 疾病名称
        self.all_herbs: List[str] = []           # 所有药食同源中药名
        self.all_diseases_cn: List[str] = []     # 可查询的中文疾病名

        self._load_and_build()

    # ===================== 数据加载 =====================

    def _load_and_build(self) -> None:
        self._cache_path = self.filepath.parent / "_graph_cache.pkl"
        # 尝试从 pickle 缓存加载
        if self._cache_path.exists() and self._cache_path.stat().st_mtime > self.filepath.stat().st_mtime:
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                for key, val in cache.items():
                    setattr(self, key, val)
                print(f"从缓存加载图谱: {len(self.all_diseases_cn)} 种疾病, {len(self.all_herbs)} 种中药")
                return
            except Exception:
                pass  # 缓存损坏，回退 Excel 加载

        xls = pd.ExcelFile(self.filepath)
        sheets = xls.sheet_names

        # 定位三个 Sheet
        herb_sheet = [s for s in sheets if "中药" in s or "herb" in s.lower()][0]
        comp_sheet = [s for s in sheets if "化合物" in s and "靶点" in s][0]
        disease_sheet = [s for s in sheets if "靶点" in s and "疾病" in s][0]

        # --- Sheet 1: 中药-化合物 ---
        self.herb_compound_df = pd.read_excel(xls, sheet_name=herb_sheet)
        self.herb_compound_df.columns = [
            "中药名", "中药ID", "化合物名", "化合物英文", "化合物ID",
            "化合物中文", "分子式", "SMILES"
        ]

        # --- Sheet 2: 化合物-靶点 ---
        self.compound_target_df = pd.read_excel(xls, sheet_name=comp_sheet)
        self.compound_target_df.columns = [
            "化合物ID", "化合物中文", "靶点ID", "靶点简称",
            "靶点全称", "靶点类型", "数据来源"
        ]

        # --- Sheet 3: 靶点-疾病 ---
        self.target_disease_df = pd.read_excel(xls, sheet_name=disease_sheet)
        self.target_disease_df.columns = [
            "靶点ID", "靶点简称", "靶点全称", "疾病ID", "疾病名称",
            "UMLS名称", "DisGeNET类型", "PubMed编号"
        ]

        self._build_indices()

        # 保存缓存供下次快速加载
        try:
            import pickle
            cache = {
                "en_to_cn": self.en_to_cn, "cn_to_en": self.cn_to_en,
                "disease_to_targets": self.disease_to_targets,
                "target_to_compounds": self.target_to_compounds,
                "compound_to_herbs": self.compound_to_herbs,
                "herb_to_compounds": self.herb_to_compounds,
                "target_names": self.target_names,
                "compound_names": self.compound_names,
                "all_diseases_cn": self.all_diseases_cn,
                "all_english_diseases": self.all_english_diseases,
                "all_diseases_cn_quality": self.all_diseases_cn_quality,
                "all_herbs": self.all_herbs,
            }
            with open(self._cache_path, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    # ===================== 构建索引 =====================

    def _build_indices(self) -> None:
        # 化合物 → 中药（从 Sheet1）
        for _, row in self.herb_compound_df.iterrows():
            herb = str(row["中药名"])
            cid = str(row["化合物ID"])
            self.compound_to_herbs.setdefault(cid, set()).add(herb)
            self.herb_to_compounds.setdefault(herb, set()).add(cid)
            self.compound_names[cid] = str(row.get("化合物中文", "") or row.get("化合物名", ""))

        self.all_herbs = sorted(self.herb_to_compounds.keys())

        # 靶点 → 化合物（从 Sheet2）
        for _, row in self.compound_target_df.iterrows():
            tid = str(row["靶点ID"])
            cid = str(row["化合物ID"])
            self.target_to_compounds.setdefault(tid, set()).add(cid)
            self.target_names[tid] = str(row.get("靶点简称", "") or row.get("靶点全称", ""))

        # 疾病 → 靶点（从 Sheet3）
        for _, row in self.target_disease_df.iterrows():
            disease = str(row["疾病名称"])
            tid = str(row["靶点ID"])
            self.disease_to_targets.setdefault(disease, set()).add(tid)
            self.disease_names[str(row["疾病ID"])] = disease
            self.target_names[tid] = str(row.get("靶点简称", "") or row.get("靶点全称", ""))

        # 建立全部英文疾病名 → 中文翻译映射（精确匹配 + 词级翻译）
        self.en_to_cn: Dict[str, str] = {}        # 英文疾病名 → 中文翻译
        self.cn_to_en: Dict[str, str] = {}         # 中文翻译 → 英文疾病名（一对一）
        for en_name in self.disease_to_targets:
            cn_name = translate_en_disease_to_cn(en_name)
            # 始终记录英文→中文（保证每个英文名都能被搜到）
            self.en_to_cn[en_name] = cn_name
            # 中文→英文：同一中文名对应多个英文时，保留靶点最多的
            if cn_name in self.cn_to_en:
                existing_en = self.cn_to_en[cn_name]
                if len(self.disease_to_targets.get(en_name, set())) > \
                   len(self.disease_to_targets.get(existing_en, set())):
                    self.cn_to_en[cn_name] = en_name
            else:
                self.cn_to_en[cn_name] = en_name

        # 查询列表
        self.all_diseases_cn = sorted(self.cn_to_en.keys())
        self.all_english_diseases = sorted(self.disease_to_targets.keys())

        # 高质量中文名（汉字占比 > 50%，默认展示，排除"英文+综合征"式拼凑）
        def _cn_ratio(name: str) -> float:
            if not name:
                return 0.0
            cn_chars = sum(1 for c in name if '一' <= c <= '鿿')
            return cn_chars / len(name)

        self.all_diseases_cn_quality = sorted(
            [n for n in self.all_diseases_cn if _cn_ratio(n) > 0.5],
            key=lambda n: (-len(self.disease_to_targets.get(self.cn_to_en[n], set())), n)
        )

    # ===================== 公开查询接口 =====================

    def resolve_cn_disease(self, cn_name: str) -> Optional[str]:
        """中文疾病名 → 新表中的英文疾病名。"""
        return CN_TO_EN_DISEASE.get(cn_name)

    def translate_cn_keywords(self, cn_text: str) -> List[str]:
        """
        将中文医学术语拆解为英文关键词，用于在英文疾病库中模糊搜索。
        示例: '生长障碍' → ['Growth', 'Disorder']
        """
        keywords = []
        remaining = cn_text
        # 先匹配长词（优先完整词组）
        for cn_term, en_term in sorted(
            CN_MEDICAL_TERMS.items(), key=lambda x: -len(x[0])
        ):
            if cn_term in remaining:
                keywords.append(en_term)
                remaining = remaining.replace(cn_term, "", 1)
        return keywords if keywords else [cn_text]

    def get_targets_by_disease(self, disease_name: str) -> Set[str]:
        """
        查询疾病关联的靶点ID集合。
        优先中文名映射，回退英文名/模糊匹配。
        """
        # 中文名 → 英文名
        en_name = self.cn_to_en.get(disease_name)
        if en_name:
            return self.disease_to_targets.get(en_name, set())
        # CN_TO_EN_DISEASE 精确映射
        en_name = CN_TO_EN_DISEASE.get(disease_name)
        if en_name and en_name in self.disease_to_targets:
            return self.disease_to_targets[en_name]
        # 直接英文名匹配
        if disease_name in self.disease_to_targets:
            return self.disease_to_targets[disease_name]
        # 模糊匹配英文名
        dl = disease_name.lower()
        for name in self.disease_to_targets:
            if dl in name.lower():
                return self.disease_to_targets[name]
        return set()

    def get_compounds_by_target(self, target_id: str) -> Set[str]:
        """查询靶点关联的化合物ID集合。"""
        return self.target_to_compounds.get(target_id, set())

    def get_herbs_by_compound(self, compound_id: str) -> Set[str]:
        """查询化合物所属的中药集合。"""
        return self.compound_to_herbs.get(compound_id, set())

    def get_compounds_by_herb(self, herb_name: str) -> Set[str]:
        """查询中药包含的化合物ID集合。"""
        return self.herb_to_compounds.get(herb_name, set())

    def get_target_name(self, target_id: str) -> str:
        return self.target_names.get(target_id, target_id)

    def get_compound_name(self, compound_id: str) -> str:
        return self.compound_names.get(compound_id, compound_id)

    # ===================== 核心：疾病 → 中药排序 =====================

    def rank_herbs_for_disease(
        self, cn_disease: str, top_k: int = 15
    ) -> List[Dict]:
        """
        一站式查询：中文疾病名 → 图谱展开 → 中药排序。

        返回按关联强度降序排列的中药列表，每项包含：
          - 中药名, 关联靶点数, 关联化合物数
          - 证据链：[(靶点名, 化合物名), ...]（按关联化合物数降序，取前5个靶点）
        """
        # 解析疾病名
        en_name = self.cn_to_en.get(cn_disease)
        if en_name is None:
            en_name = CN_TO_EN_DISEASE.get(cn_disease)
        if en_name is None and cn_disease in self.disease_to_targets:
            en_name = cn_disease
        if en_name is None:
            dl = cn_disease.lower()
            for name in self.disease_to_targets:
                if dl in name.lower():
                    en_name = name
                    break
        if en_name is None:
            return []

        # 1. 疾病 → 靶点
        target_ids = self.disease_to_targets.get(en_name, set())
        if not target_ids:
            return []

        # 2. 靶点 → 化合物
        all_compound_ids: Set[str] = set()
        target_to_comps: Dict[str, Set[str]] = {}
        for tid in target_ids:
            comps = self.target_to_compounds.get(tid, set())
            if comps:
                all_compound_ids.update(comps)
                target_to_comps[tid] = comps

        if not all_compound_ids:
            return []

        # 3. 化合物 → 中药：统计每味中药下每个靶点关联的化合物数
        herb_stats: Dict[str, Dict] = {}  # herb → {targets, compounds, target_compounds}

        for tid, comps in target_to_comps.items():
            tname = self.target_names.get(tid, tid)
            for cid in comps:
                cname = self.compound_names.get(cid, cid)
                herbs = self.compound_to_herbs.get(cid, set())
                for herb in herbs:
                    if herb not in herb_stats:
                        herb_stats[herb] = {
                            "targets": set(),
                            "compounds": set(),
                            "target_compounds": {},  # tid -> set of cnames
                        }
                    herb_stats[herb]["targets"].add(tid)
                    herb_stats[herb]["compounds"].add(cid)
                    if tid not in herb_stats[herb]["target_compounds"]:
                        herb_stats[herb]["target_compounds"][tid] = set()
                    herb_stats[herb]["target_compounds"][tid].add(cname)

        # 4. 构建证据链 + 排序
        ranked = []
        for herb, stats in herb_stats.items():
            # 按每个靶点关联的化合物数量降序，取前5个最强靶点
            tc = stats["target_compounds"]
            sorted_targets = sorted(tc.items(), key=lambda x: len(x[1]), reverse=True)
            evidence = []
            for tid, cnames in sorted_targets:
                tname = self.target_names.get(tid, tid)
                evidence.append((tname, next(iter(cnames))))
                if len(evidence) >= 5:
                    break

            ranked.append({
                "中药名": herb,
                "关联靶点数": len(stats["targets"]),
                "关联化合物数": len(stats["compounds"]),
                "证据链": evidence,
            })

        ranked.sort(key=lambda x: (x["关联靶点数"], x["关联化合物数"]), reverse=True)
        return ranked[:top_k]

    # ===================== 图谱统计 =====================

    def get_graph_stats(self, cn_disease: str) -> Dict:
        """获取某个疾病在图谱中的统计数据。"""
        en_name = self.cn_to_en.get(cn_disease)
        if en_name is None:
            en_name = CN_TO_EN_DISEASE.get(cn_disease)
        if en_name is None and cn_disease in self.disease_to_targets:
            en_name = cn_disease
        if en_name is None:
            dl = cn_disease.lower()
            for name in self.disease_to_targets:
                if dl in name.lower():
                    en_name = name
                    break
        if en_name is None:
            return {}

        target_ids = self.disease_to_targets.get(en_name, set())
        compound_ids: Set[str] = set()
        herb_names: Set[str] = set()

        for tid in target_ids:
            for cid in self.target_to_compounds.get(tid, set()):
                compound_ids.add(cid)
                herb_names.update(self.compound_to_herbs.get(cid, set()))

        return {
            "疾病英文名": en_name,
            "关联靶点数": len(target_ids),
            "关联化合物数": len(compound_ids),
            "相关中药数": len(herb_names),
        }

    def get_herb_disease_chain(
        self, herb_name: str, disease_cn: str,
        max_ingredients: int = 8, max_genes: int = 12,
    ) -> Dict:
        """
        获取指定中药与疾病之间的分子关联链路。
        返回可用于绘制环形知识图谱的化合物、靶点列表及映射关系。

        Returns:
            {
                "compounds": [(cid, cname), ...],   # 化合物ID+中文名
                "targets": [(tid, tname), ...],      # 靶点ID+简称
                "compound_target_map": {cid: [(tid, tname), ...]},  # 化合物→靶点
            }
        """
        # 解析疾病英文名
        en_disease = self.cn_to_en.get(disease_cn)
        if en_disease is None:
            en_disease = CN_TO_EN_DISEASE.get(disease_cn)
        if en_disease is None and disease_cn in self.disease_to_targets:
            en_disease = disease_cn
        if en_disease is None:
            dl = disease_cn.lower()
            for name in self.disease_to_targets:
                if dl in name.lower():
                    en_disease = name
                    break
        if en_disease is None:
            return {"compounds": [], "targets": [], "compound_target_map": {}}

        # 疾病关联的靶点
        disease_targets = self.disease_to_targets.get(en_disease, set())
        if not disease_targets:
            return {"compounds": [], "targets": [], "compound_target_map": {}}

        # 该中药含有的化合物
        herb_compounds = self.herb_to_compounds.get(herb_name, set())
        if not herb_compounds:
            return {"compounds": [], "targets": [], "compound_target_map": {}}

        # pandas 向量化筛选：化合物在 herb 中 且 靶点在 disease 中
        mask = (
            self.compound_target_df["化合物ID"].isin(herb_compounds)
            & self.compound_target_df["靶点ID"].isin(disease_targets)
        )
        matched = self.compound_target_df[mask]

        if matched.empty:
            return {"compounds": [], "targets": [], "compound_target_map": {}}

        # 按靶点出现频次对化合物排序，截断
        compound_target_count = matched.groupby("化合物ID").size().sort_values(ascending=False)
        top_compounds = compound_target_count.head(max_ingredients).index.tolist()

        # 收集这些化合物关联的靶点，截断
        matched_top = matched[matched["化合物ID"].isin(top_compounds)]
        target_compound_count = matched_top.groupby("靶点ID").size().sort_values(ascending=False)
        top_targets = target_compound_count.head(max_genes).index.tolist()

        # 构建返回结果
        compound_target_map = {}
        for cid in top_compounds:
            c_targets = matched_top[matched_top["化合物ID"] == cid]["靶点ID"].tolist()
            c_targets = [t for t in c_targets if t in top_targets]
            if c_targets:
                compound_target_map[cid] = [
                    (t, self.target_names.get(t, t)) for t in c_targets
                ]

        return {
            "compounds": [(cid, self.compound_names.get(cid, cid)) for cid in top_compounds if cid in compound_target_map],
            "targets": [(tid, self.target_names.get(tid, tid)) for tid in top_targets],
            "compound_target_map": compound_target_map,
        }

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "知识图谱数据概览",
            "=" * 50,
            f"药食同源中药: {len(self.all_herbs)} 种",
            f"化合物: {len(self.compound_names)} 种",
            f"靶点: {len(self.target_names)} 个",
            f"可查询中文疾病: {len(self.all_diseases_cn)} 种",
            f"中药-化合物关系: {len(self.herb_compound_df)} 条",
            f"化合物-靶点关系: {len(self.compound_target_df)} 条",
            f"靶点-疾病关系: {len(self.target_disease_df)} 条",
            "=" * 50,
        ]
        return "\n".join(lines)


# ===================== 模块自测 =====================
if __name__ == "__main__":
    loader = GraphDataLoader()
    print(loader.summary())

    for disease in ["糖尿病", "贫血", "痛风"]:
        stats = loader.get_graph_stats(disease)
        print(f"\n{disease} ({stats.get('疾病英文名', 'N/A')}):")
        print(f"  靶点: {stats['关联靶点数']}, "
              f"化合物: {stats['关联化合物数']}, "
              f"中药: {stats['相关中药数']}")

        herbs = loader.rank_herbs_for_disease(disease, top_k=5)
        for i, h in enumerate(herbs, 1):
            print(f"  {i}. {h['中药名']} (靶点:{h['关联靶点数']}, 化合物:{h['关联化合物数']})")
