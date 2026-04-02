from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "medical_texts" / "medical_corpus.jsonl"


DISEASES = [
    {
        "name": "感冒",
        "aliases": ["普通感冒", "上呼吸道感染"],
        "department": "呼吸内科",
        "symptoms": ["流涕", "鼻塞", "咽痛", "低热"],
        "complications": ["鼻窦炎", "中耳炎"],
        "drugs": [("对乙酰氨基酚", "解热镇痛药"), ("氯雷他定", "抗组胺药")],
        "exams": ["血常规", "体格检查"],
        "risk_factors": ["受凉", "熬夜", "免疫力下降"],
        "pathogens": ["鼻病毒", "冠状病毒"],
    },
    {
        "name": "流感",
        "aliases": ["流行性感冒"],
        "department": "感染科",
        "symptoms": ["高热", "头痛", "肌肉酸痛", "乏力"],
        "complications": ["肺炎", "心肌炎"],
        "drugs": [("奥司他韦", "抗病毒药"), ("对乙酰氨基酚", "解热镇痛药")],
        "exams": ["甲型乙型流感抗原检测", "血常规"],
        "risk_factors": ["未接种疫苗", "人群密集接触", "免疫力下降"],
        "pathogens": ["甲型流感病毒", "乙型流感病毒"],
    },
    {
        "name": "肺炎",
        "aliases": ["社区获得性肺炎"],
        "department": "呼吸内科",
        "symptoms": ["发热", "咳嗽", "咳痰", "胸痛"],
        "complications": ["脓毒症", "胸腔积液"],
        "drugs": [("阿莫西林", "抗生素"), ("左氧氟沙星", "抗生素"), ("布洛芬", "解热镇痛药")],
        "exams": ["胸部CT", "血常规", "痰培养"],
        "risk_factors": ["高龄", "吸烟", "慢性肺病"],
        "pathogens": ["肺炎链球菌", "流感嗜血杆菌"],
    },
    {
        "name": "急性支气管炎",
        "aliases": ["支气管炎"],
        "department": "呼吸内科",
        "symptoms": ["咳嗽", "咳痰", "发热", "胸闷"],
        "complications": ["肺炎", "喘息加重"],
        "drugs": [("氨溴索", "祛痰药"), ("右美沙芬", "镇咳药"), ("阿莫西林", "抗生素")],
        "exams": ["胸片", "血常规"],
        "risk_factors": ["吸烟", "空气污染", "病毒感染"],
        "pathogens": ["呼吸道合胞病毒", "肺炎支原体"],
    },
    {
        "name": "慢阻肺急性加重",
        "aliases": ["AECOPD", "慢性阻塞性肺疾病急性加重"],
        "department": "呼吸内科",
        "symptoms": ["气促", "咳痰增多", "喘息", "胸闷"],
        "complications": ["呼吸衰竭", "肺部感染"],
        "drugs": [("沙丁胺醇", "支气管扩张剂"), ("布地奈德", "糖皮质激素"), ("莫西沙星", "抗生素")],
        "exams": ["肺功能检查", "血气分析", "胸部CT"],
        "risk_factors": ["长期吸烟", "空气污染", "反复感染"],
        "pathogens": ["流感嗜血杆菌", "肺炎链球菌"],
    },
    {
        "name": "扁桃体炎",
        "aliases": ["急性扁桃体炎"],
        "department": "耳鼻喉科",
        "symptoms": ["咽痛", "发热", "吞咽困难", "扁桃体肿大"],
        "complications": ["中耳炎", "风湿热"],
        "drugs": [("阿莫西林", "抗生素"), ("头孢克肟", "抗生素"), ("布洛芬", "解热镇痛药")],
        "exams": ["咽拭子培养", "血常规"],
        "risk_factors": ["受凉", "细菌感染", "免疫力下降"],
        "pathogens": ["A组链球菌", "葡萄球菌"],
    },
    {
        "name": "中耳炎",
        "aliases": ["急性中耳炎"],
        "department": "耳鼻喉科",
        "symptoms": ["耳痛", "听力下降", "发热", "耳鸣"],
        "complications": ["乳突炎", "鼓膜穿孔"],
        "drugs": [("阿莫西林克拉维酸钾", "抗生素"), ("左氧氟沙星滴耳液", "抗生素")],
        "exams": ["耳镜检查", "听力学检查"],
        "risk_factors": ["上呼吸道感染", "咽鼓管功能障碍", "儿童期"],
        "pathogens": ["肺炎链球菌", "流感嗜血杆菌"],
    },
    {
        "name": "尿路感染",
        "aliases": ["泌尿系感染"],
        "department": "泌尿外科",
        "symptoms": ["尿频", "尿急", "尿痛", "下腹痛"],
        "complications": ["肾盂肾炎", "脓毒症"],
        "drugs": [("左氧氟沙星", "抗生素"), ("头孢呋辛", "抗生素"), ("磷霉素", "抗生素")],
        "exams": ["尿常规", "尿培养", "泌尿系超声"],
        "risk_factors": ["女性解剖结构", "饮水少", "导尿操作"],
        "pathogens": ["大肠埃希菌", "肺炎克雷伯菌"],
    },
    {
        "name": "胃溃疡",
        "aliases": ["消化性溃疡"],
        "department": "消化内科",
        "symptoms": ["上腹痛", "反酸", "恶心", "腹胀"],
        "complications": ["消化道出血", "穿孔"],
        "drugs": [("奥美拉唑", "抑酸药"), ("枸橼酸铋钾", "胃黏膜保护剂"), ("阿莫西林", "抗生素")],
        "exams": ["胃镜", "幽门螺杆菌检测"],
        "risk_factors": ["幽门螺杆菌感染", "长期服用NSAIDs", "吸烟"],
        "pathogens": ["幽门螺杆菌"],
    },
    {
        "name": "高血压",
        "aliases": ["原发性高血压"],
        "department": "心血管内科",
        "symptoms": ["头晕", "头痛", "心悸", "乏力"],
        "complications": ["脑卒中", "冠心病"],
        "drugs": [("氨氯地平", "降压药"), ("缬沙坦", "降压药")],
        "exams": ["动态血压监测", "心电图", "肾功能检查"],
        "risk_factors": ["高盐饮食", "肥胖", "家族史"],
        "pathogens": [],
    },
    {
        "name": "糖尿病",
        "aliases": ["2型糖尿病"],
        "department": "内分泌科",
        "symptoms": ["多饮", "多尿", "体重下降", "乏力"],
        "complications": ["糖尿病肾病", "糖尿病足"],
        "drugs": [("二甲双胍", "降糖药"), ("胰岛素", "降糖药")],
        "exams": ["空腹血糖", "糖化血红蛋白", "尿微量白蛋白"],
        "risk_factors": ["肥胖", "久坐", "家族史"],
        "pathogens": [],
    },
    {
        "name": "冠心病",
        "aliases": ["冠状动脉粥样硬化性心脏病"],
        "department": "心血管内科",
        "symptoms": ["胸痛", "胸闷", "气短", "心悸"],
        "complications": ["心肌梗死", "心力衰竭"],
        "drugs": [("阿司匹林", "抗血小板药"), ("阿托伐他汀", "调脂药"), ("硝酸甘油", "扩血管药")],
        "exams": ["冠脉CT", "心电图", "肌钙蛋白"],
        "risk_factors": ["高血压", "高脂血症", "吸烟"],
        "pathogens": [],
    },
    {
        "name": "哮喘",
        "aliases": ["支气管哮喘"],
        "department": "呼吸内科",
        "symptoms": ["喘息", "气促", "咳嗽", "胸闷"],
        "complications": ["呼吸衰竭", "肺部感染"],
        "drugs": [("沙丁胺醇", "支气管扩张剂"), ("布地奈德", "糖皮质激素"), ("孟鲁司特", "抗炎药")],
        "exams": ["肺功能检查", "过敏原检测"],
        "risk_factors": ["过敏体质", "尘螨暴露", "冷空气刺激"],
        "pathogens": [],
    },
    {
        "name": "阑尾炎",
        "aliases": ["急性阑尾炎"],
        "department": "普外科",
        "symptoms": ["右下腹痛", "恶心", "发热", "食欲减退"],
        "complications": ["腹膜炎", "脓肿形成"],
        "drugs": [("头孢曲松", "抗生素"), ("甲硝唑", "抗生素")],
        "exams": ["腹部超声", "腹部CT", "血常规"],
        "risk_factors": ["粪石堵塞", "阑尾腔梗阻", "细菌感染"],
        "pathogens": ["大肠埃希菌", "厌氧菌"],
    },
    {
        "name": "胆囊炎",
        "aliases": ["急性胆囊炎"],
        "department": "普外科",
        "symptoms": ["右上腹痛", "发热", "恶心", "呕吐"],
        "complications": ["胆囊穿孔", "腹膜炎"],
        "drugs": [("头孢哌酮舒巴坦", "抗生素"), ("甲硝唑", "抗生素")],
        "exams": ["腹部超声", "肝功能检查", "血常规"],
        "risk_factors": ["胆结石", "高脂饮食", "肥胖"],
        "pathogens": ["大肠埃希菌", "肠球菌"],
    },
    {
        "name": "脓毒症",
        "aliases": ["败血症", "感染性休克前期"],
        "department": "重症医学科",
        "symptoms": ["高热", "寒战", "低血压", "意识改变"],
        "complications": ["感染性休克", "多器官功能障碍"],
        "drugs": [("美罗培南", "抗生素"), ("哌拉西林他唑巴坦", "抗生素"), ("万古霉素", "抗生素")],
        "exams": ["血培养", "降钙素原", "乳酸检测"],
        "risk_factors": ["严重感染", "免疫抑制", "侵入性操作"],
        "pathogens": ["金黄色葡萄球菌", "大肠埃希菌"],
    },
]


TEMPLATES = [
    "概述：{name}是一种常见的{department}疾病。别名包括{aliases}。常见症状包括{symptoms}。常见并发症包括{complications}。",
    "治疗要点：{name}的常用药物包括{drug_names}。其中属于抗生素的药物包括{antibiotics}。推荐检查包括{exams}。",
    "风险提示：{name}的高危因素包括{risk_factors}。常见致病因素或病原体包括{pathogens}。若控制不佳，常见并发症包括{complications}。",
    "门诊宣教：医生通常根据{symptoms}判断{name}的临床表现，再结合{exams}进行评估。所属科室为{department}。",
    "护理建议：针对{name}，患者常诉{symptoms}。治疗阶段可用{drug_names}，并警惕{complications}等问题。",
    "病例摘要：患者因{symptoms}就诊，初步考虑{name}。进一步建议完成{exams}。若出现{complications}，常需强化治疗。",
    "用药说明：{name}治疗常涉及{drug_names}。当怀疑细菌感染时，适用抗生素包括{antibiotics}。别名包括{aliases}。",
    "复习笔记：{name}属于{department}常见病。高危因素包括{risk_factors}。推荐检查包括{exams}。常见并发症包括{complications}。",
]


def join_items(items: list[str], empty_text: str = "暂无明确记录") -> str:
    return "、".join(items) if items else empty_text


def build_documents() -> list[dict]:
    documents: list[dict] = []
    doc_index = 1
    for disease in DISEASES:
        aliases = join_items(disease["aliases"])
        symptoms = join_items(disease["symptoms"])
        complications = join_items(disease["complications"])
        drug_names = join_items([item[0] for item in disease["drugs"]])
        antibiotics = join_items([name for name, cls in disease["drugs"] if cls == "抗生素"], "暂无明确抗生素")
        exams = join_items(disease["exams"])
        risk_factors = join_items(disease["risk_factors"])
        pathogens = join_items(disease["pathogens"])

        for template in TEMPLATES:
            for variant in range(4):
                text = template.format(
                    name=disease["name"],
                    aliases=aliases,
                    symptoms=symptoms,
                    complications=complications,
                    drug_names=drug_names,
                    antibiotics=antibiotics,
                    exams=exams,
                    risk_factors=risk_factors,
                    pathogens=pathogens,
                    department=disease["department"],
                )
                text = f"{text} 第{variant + 1}版临床整理，供课程项目知识抽取与检索实验使用。"
                documents.append(
                    {
                        "doc_id": f"MED-{doc_index:04d}",
                        "title": f"{disease['name']}-知识条目-{variant + 1}",
                        "category": "medical_encyclopedia",
                        "source": "synthetic_medical_guide",
                        "disease": disease["name"],
                        "text": text,
                    }
                )
                doc_index += 1
    return documents


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    documents = build_documents()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in documents:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Generated {len(documents)} documents at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
