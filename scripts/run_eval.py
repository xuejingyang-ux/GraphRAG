# -*- coding: utf-8 -*-
import json
import os
import subprocess
import time
from urllib import request


SAMPLES = [
    "周星驰，1962年6月22日出生于中国香港，祖籍浙江宁波，是演员、导演、编剧、制作人。",
    "《大话西游》由刘镇伟执导，周星驰、朱茵、吴孟达等主演，是华语经典奇幻喜剧电影。",
    "刘镇伟，英文名 Jeffrey Lau，是中国香港导演、编剧、制作人。",
    "《功夫》由周星驰执导并主演，是一部融合动作与喜剧风格的电影。",
    "星爷通常被认为是周星驰的常见别称。",
    "《少林足球》由周星驰自编自导自演，将功夫元素与足球运动结合。",
    "吴孟达是周星驰的重要合作伙伴，两人共同出演过多部经典电影。",
    "朱茵在《大话西游》中饰演紫霞仙子，这一角色广受欢迎。",
    "西安电影制片厂成立于1958年，是中国重要的电影制片机构之一。",
    "丽的电视后来更名为亚洲电视，简称 ATV。",
]

TEST_CASES = [
    ("简单查询", "《功夫》是谁执导的？"),
    ("多跳推理", "紫霞仙子这一角色和周星驰有什么关系？"),
    ("边界案例", "周星驰获得过几次奥斯卡奖？"),
]


def post(base: str, path: str, data: dict) -> dict:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        base + path,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(base: str, path: str) -> dict:
    with request.urlopen(base + path, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PORT"] = "3001"
    env["DISABLE_HMR"] = "true"
    proc = subprocess.Popen(
        ["node", "server.mjs"],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    base = "http://localhost:3001"
    try:
        started = False
        start = time.time()
        logs: list[str] = []
        while time.time() - start < 20:
            line = proc.stdout.readline()
            if line:
                logs.append(line.rstrip())
                if "Server running on http://localhost:3001" in line:
                    started = True
                    break
            else:
                time.sleep(0.2)

        if not started:
            raise RuntimeError("Server did not start.\n" + "\n".join(logs[-20:]))

        post(base, "/api/clear", {})
        ingest_results = [post(base, "/api/ingest", {"text": text}) for text in SAMPLES]

        results = []
        for name, query in TEST_CASES:
            vector = post(base, "/api/query-vector", {"query": query})
            graph = post(base, "/api/query", {"query": query})
            results.append(
                {
                    "name": name,
                    "query": query,
                    "vector": vector,
                    "graph": graph,
                }
            )

        payload = {
            "ingest_results": ingest_results,
            "graph_snapshot": get(base, "/api/graph"),
            "results": results,
        }

        out_path = os.path.join(workdir, "test_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(out_path)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
