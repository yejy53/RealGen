import os
import json
import base64
import io
from PIL import Image
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import threading

# ==== 1. 批量配置 (在这里修改) ====

# 存放所有JSONL输出的基础目录
BASE_OUTPUT_DIR = ""

# (新) 待分析的文件夹列表 (包括真实图片和AI生成的图片)
ALL_DIRS_TO_ANALYZE = [
    # 你的AI生成图片文件夹
    
]

# ==== 2. API 和线程配置 (保持不变) ====
client = OpenAI(
    base_url="",
    api_key=""
)

write_lock = threading.Lock()
MAX_WORKERS = 200

# ==== 3. 新的 Prompt 构建函数 (来自你的确认) ====
def build_prompt(image_path, base64_str):
    """
    构建用于图像真实性检测的Prompt。
    
    最终优化版：
    1. 确保评估的客观、严格、公正，消除了诱导性偏见。
    2. 增加了非人像的检测标准（物理、文字、纹理）。
    3. 专门增加了一条针对人像（特别是皮肤油腻感）的检测标准。
    """
    # image_path 在此函数中未被使用，但保留签名以匹配旧用法
    return [
        {"role": "user", "content": """Assume you are an advanced **forensic image analyst** and AI detection expert. Your task is to conduct an **objective, rigorous, and strictly impartial evaluation** of the following image to determine if it is **real (photographic) or AI-generated (fake)**.

**You must apply a high standard of scrutiny to *all* images.** Do not default to 'Real' or 'Fake'; your conclusion must be based *only* on the evidence present in the image.

Consider the following criteria for your forensic judgment:

1.  **Physical & Geometric Logic:** Analyze physical interactions. Are shadows cast correctly according to light sources? Are reflections plausible and consistent? Do objects rest on surfaces naturally? Scrutinize architecture and rigid objects for warped lines, impossible geometry, or inconsistent perspectives.
2.  **Text, Symbols, and Fine Detail:** Examine any text, logos, or signs. AI-generated text is often nonsensical, misspelled, or has warped/merged characters. This is a very strong indicator.
3.  **General Texture & Surface Coherence:** Look at surfaces like wood, metal, water, or fabric. AI often produces textures that are overly smooth, blurry, lack fine natural detail (like wood grain), or have strange, repetitive patterns.
4.  **Human & Animal Subjects (if present):** If figures are present, scrutinize them for specific AI artifacts. Look for:
    * Distorted anatomical features (e.g., malformed hands, incorrect number of fingers, unnatural eyes/ears).
    * **Unnatural skin texture**, such as an **overly smooth, "greasy," or "plastic-like" sheen** that lacks natural pores and imperfections.
5.  **Lighting & Color Consistency:** Is the lighting consistent across the entire image? Do different elements look like they belong in the same environment? Look for unnatural color bleed or overly synthetic, "flat" saturation.
6.  **Edges & Artifacts:** Check the edges where objects meet. Look for unnatural sharpness, strange blurring (artifacting), or a 'cut-out' appearance that doesn't match a natural depth of field.

Please output your result in the *exact* following format:
Real or Fake: [Choose one]
Reason: [Provide a concise, objective explanation for your judgment, *directly referencing the specific criteria* (e.g., "Text is warped," "Skin texture appears overly plastic," "Shadows are physically inconsistent") that led to your conclusion.]
"""},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}
        ]}
    ]

# ==== 4. 工具函数 (已更新) ====
def encode_image_to_base64(image_path):
    """将单个图像文件编码为Base64。"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def load_jsonl_records(path: str):
    """
    加载JSONL记录以进行断点续传。
    (已更新：现在检查 'image_name' 而不是 'image_pair')
    """
    records = []
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                    if "image_name" in rec:
                        done.add(rec["image_name"])
                except Exception:
                    continue
    return records, done

# ==== 5. 核心分析逻辑 (新函数) ====
def process_single_image(image_path, max_retries=5, backoff_base=1.0):
    """
    处理单个图像并返回AI的 (Real/Fake) 判断结果。
    """
    fname = os.path.basename(image_path)
    
    try:
        base64_str = encode_image_to_base64(image_path)
        messages = build_prompt(image_path, base64_str)
    except Exception as e:
        return {"image_name": fname, "image_path": image_path, "response": None, "error": f"Image encode error: {e}"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=messages
            )
            content = response.choices[0].message.content
            response_text = (content or "").strip()

            if response_text:
                return {"image_name": fname, "image_path": image_path, "response": response_text}
            raise RuntimeError("empty response")

        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                sleep_s = backoff_base * (2 ** (attempt - 1)) + random.random() * 0.5
                time.sleep(sleep_s)
                continue
            return {"image_name": fname, "image_path": image_path, "response": None, "error": last_err or "Max retries exceeded"}

    return {"image_name": fname, "image_path": image_path, "response": None, "error": last_err or "Unknown error"}

def run_analysis_on_directory(image_dir, output_jsonl):
    """
    运行一次完整的分析流程。
    Args:
        image_dir (str): 图像文件夹路径。
        output_jsonl (str): 此次分析的结果 .jsonl 文件路径。
    """
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    records, done_images = load_jsonl_records(output_jsonl)

    # 1. 查找文件
    try:
        all_files = {f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    except FileNotFoundError as e:
        print(f"\n[Error] 无法找到文件夹: {e.filename}。跳过此次分析。")
        return

    all_paths = sorted([os.path.join(image_dir, f) for f in all_files])
    to_process = [p for p in all_paths if os.path.basename(p) not in done_images]
    
    print(f"  > 找到 {len(all_paths)} 个图像。")
    print(f"  > ✅ 已完成: {len(done_images)} | 🚀 待处理: {len(to_process)}")

    # (新) 统计标准
    counts = {"Real": 0, "Fake": 0, "Unknown/Error": 0}
    
    # (新) 从已有的记录中恢复计数
    for rec in records:
        response_lower = rec.get("response", "").lower()
        if "real or fake: real" in response_lower:
            counts["Real"] += 1
        elif "real or fake: fake" in response_lower:
            counts["Fake"] += 1
        else:
            counts["Unknown/Error"] += 1

    if not to_process:
        print("  > 无需处理，跳至下一个。")
        # 即使无需处理，也打印已有的统计结果
        print(f"\n  --- 📊 分析总结 ({os.path.basename(image_dir)}) ---")
        print(f"  判断为 Real: {counts['Real']} 次")
        print(f"  判断为 Fake: {counts['Fake']} 次")
        print(f"  无法判断 / 出现错误: {counts['Unknown/Error']} 次")
        return

    # 2. 多线程处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(output_jsonl, "a", encoding="utf-8") as f:
        futures = {executor.submit(process_single_image, img_path): img_path for img_path in to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Analyzing {os.path.basename(image_dir)}"):
            img_path = futures[future]
            try:
                result = future.result()
                response_text = result.get("response")

                if response_text:
                    with write_lock:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

                    response_lower = response_text.lower()
                    if "real or fake: real" in response_lower:
                        counts["Real"] += 1
                    elif "real or fake: fake" in response_lower:
                        counts["Fake"] += 1
                    else:
                        counts["Unknown/Error"] += 1
                else:
                    counts["Unknown/Error"] += 1
                    print(f"\n[Warning] 图像 {os.path.basename(img_path)} 处理出错: {result.get('error')}")

            except Exception as e:
                counts["Unknown/Error"] += 1
                print(f"\n[Critical] 任务 {os.path.basename(img_path)} 失败: {e}")

    print(f"\n  > ✅ 全部完成。 分析结果已保存到 -> {output_jsonl}")

    # 3. 打印总结
    print(f"\n  --- 📊 分析总结 ({os.path.basename(image_dir)}) ---")
    print(f"  判断为 Real: {counts['Real']} 次")
    print(f"  判断为 Fake: {counts['Fake']} 次")
    print(f"  无法判断 / 出现错误: {counts['Unknown/Error']} 次")


# ==== 6. 主控制器 (已修复) ====
def main():
    """
    主控制器，循环执行所有分析任务。
    """
    print("===== 🤖 开始批量图像分析任务 =====")
    print(f"输出目录: {BASE_OUTPUT_DIR}")
    print(f"总任务数: {len(ALL_DIRS_TO_ANALYZE)}")
    print("---------------------------------")

    for i, image_dir in enumerate(ALL_DIRS_TO_ANALYZE):
        
        # --- (BUG 修复开始) ---
        try:
            # 移除路径末尾可能存在的斜杠 (e.g., /.../gpt/photo/ -> /.../gpt/photo)
            clean_path = image_dir.rstrip(os.path.sep)
            
            # 获取路径的最后一部分
            last_part = os.path.basename(clean_path)
            
            if last_part.lower() == "photo":
                # 如果最后是 'photo', 我们取它上一级的目录名
                # e.g., /.../gpt/photo -> "gpt"
                model_name = os.path.basename(os.path.dirname(clean_path))
            else:
                # 否则，我们直接使用最后一部分
                # e.g., /.../real-img-banchmark -> "real-img-banchmark"
                model_name = last_part
        
        except Exception:
            # 备用方案，以防路径非常奇怪
            model_name = f"task_{i}"
        # --- (BUG 修复结束) ---
            
        # 自动生成输出文件名
        output_file_path = os.path.join(BASE_OUTPUT_DIR, f"analysis_{model_name}.jsonl")

        print(f"\n===== 🚀 [任务 {i+1}/{len(ALL_DIRS_TO_ANALYZE)}] 开始分析 =====")
        print(f"  分析目录: {model_name}")
        print(f"  完整路径: {image_dir}")
        print(f"  输出文件:   {output_file_path}")
        print("---------------------------------")

        # 执行单次分析
        run_analysis_on_directory(
            image_dir=image_dir,
            output_jsonl=output_file_path
        )

        print(f"===== ✅ [任务 {i+1}/{len(ALL_DIRS_TO_ANALYZE)}] {model_name} 分析完成 =====")

    print("\n===== 🎉 所有批量任务均已完成! =====")

if __name__ == "__main__":
    main()