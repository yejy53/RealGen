import os
import json
import base64
import io
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from openai import OpenAI
from tqdm import tqdm


# ==========================================
# 1. Configuration (用户配置区)
# ==========================================

# 固定的基准文件夹 (Image 2 - 通常是真实图片或基准模型生成的图片)
BENCHMARK_DIR = "./benchmark/real-img-benchmark"

# 存放所有 JSONL 对比结果的基础目录
BASE_OUTPUT_DIR = "./eval-results/"

# 待测试的模型图片文件夹列表 (Image 1 - 测试组)
# 对比图片的命名需要和基准命名对应，prompt路径参考"./eval/prompt.txt"，图片命名第一行为00000.jpg以此类推
DIRS_TO_TEST = [
    "./your/image/path/photo",
    "./your/image/path/photo"
]

# 线程数配置
MAX_WORKERS = 1


# ==========================================
# 2. API Setup (API 及线程锁初始化)
# ==========================================

# 推荐通过环境变量配置 API Key，避免在开源代码中泄露隐私
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# 用于保护多线程写入文件时的线程锁
write_lock = threading.Lock()


# ==========================================
# 3. Utility Functions (工具函数)
# ==========================================

def encode_image_to_base64(image_path: str) -> str:
    """将单个图像文件读取并编码为 Base64 字符串。"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def build_prompt(base64_str_1: str, base64_str_2: str) -> list:
    """构建包含两张图片的 API 提示词(Prompt)。"""
    prompt_text = """Assume you are an advanced image detection expert with deep knowledge of image analysis and AI-generated content recognition. You are given two images. Please evaluate both images and determine which one is more realistic, using the following criteria:
1. Unnatural details in human features:
   - Hands: Examine both images for any unnatural or distorted features in the hands (e.g., extra fingers, awkward positioning, or unnatural proportions).
   - Faces: Pay close attention to the faces in both images. Look for distorted facial features (e.g., misaligned eyes, overly smooth skin, unrealistic lighting on the face, or unnatural facial expressions).
2. Repetitive elements or artificial patterns: Check for background elements that seem repetitive, unnatural, or artificial.
3. Lighting: Assess the lighting in each image, especially in relation to the hands, faces, and backgrounds. AI-generated images may exhibit overly smooth or artificial lighting.
4. Discrepancies in edges: Evaluate the edges of the hands, faces, and background. 
5. Overall style and consistency: Assess the overall color tones, contrast, and saturation.

Now, please compare the two images and choose the one that is more realistic based on the above criteria.
Output format:
More Realistic Image: [Choose one image, Image 1 or Image 2]
Reason: [Provide a concise explanation for your judgment, clearly indicating which image appeared more realistic and why, based on the above criteria, with a focus on hands, faces, and the background.]
Ensure consistency in your output format as described."""

    return [
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": [
            {"type": "text", "text": "Image 1:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str_1}"}},
            {"type": "text", "text": "Image 2:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str_2}"}}
        ]}
    ]

def load_jsonl_records(path: str):
    """加载已有的 JSONL 记录，用于实现断点续传。"""
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
                    if "image_pair" in rec:
                        done.add(rec["image_pair"])
                except Exception:
                    continue
    return records, done

def process_image_pair(image_path_1: str, image_path_2: str, max_retries: int = 5, backoff_base: float = 1.0) -> dict:
    """处理一对图像，调用大模型API并返回比较结果（包含指数退避重试机制）。"""
    fname = os.path.basename(image_path_1)
    
    try:
        base64_str_1 = encode_image_to_base64(image_path_1)
        base64_str_2 = encode_image_to_base64(image_path_2)
        messages = build_prompt(base64_str_1, base64_str_2)
    except Exception as e:
        return {"image_pair": fname, "response": None, "error": f"Image encode error: {e}"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-5", # 请根据实际使用的模型名称修改 (原代码为 gpt-5)
                messages=messages
            )
            content = response.choices[0].message.content
            response_text = (content or "").strip()

            if response_text:
                return {"image_pair": fname, "response": response_text}
            raise RuntimeError("Empty response from API")

        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                # 指数退避策略 (Exponential Backoff) + 随机抖动 (Jitter)
                sleep_s = backoff_base * (2 ** (attempt - 1)) + random.random() * 0.5
                time.sleep(sleep_s)
                continue
            return {"image_pair": fname, "response": None, "error": last_err or "Max retries exceeded"}

    return {"image_pair": fname, "response": None, "error": last_err or "Unknown error"}


# ==========================================
# 4. Core Logic (核心并发比较逻辑)
# ==========================================

def run_comparison(image_dir_1: str, image_dir_2: str, output_jsonl: str):
    """
    运行一次完整的两组文件夹图片比较流程。
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    # 获取已处理记录，实现断点续传
    records, done_images = load_jsonl_records(output_jsonl)

    # 1. 查找有效的文件对
    try:
        valid_extensions = (".jpg", ".jpeg", ".png")
        all_files_1 = {f for f in os.listdir(image_dir_1) if f.lower().endswith(valid_extensions)}
        all_files_2 = {f for f in os.listdir(image_dir_2) if f.lower().endswith(valid_extensions)}
    except FileNotFoundError as e:
        print(f"\n[Error] 无法找到文件夹: {e.filename}。跳过此次比较。")
        return

    common_files = sorted(list(all_files_1.intersection(all_files_2)))
    all_pairs = [(os.path.join(image_dir_1, f), os.path.join(image_dir_2, f)) for f in common_files]
    to_process = [(p1, p2) for p1, p2 in all_pairs if os.path.basename(p1) not in done_images]
    
    print(f"  > 找到 {len(all_pairs)} 个同名图像对。")
    print(f"  > ✅ 已完成: {len(done_images)} | 🚀 待处理: {len(to_process)}")

    # 2. 统计结果初始化 (从已有记录中恢复计数)
    counts = {"Image 1": 0, "Image 2": 0, "Unknown/Error": 0}
    for rec in records:
        response_lower = rec.get("response", "").lower()
        if "more realistic image: image 1" in response_lower:
            counts["Image 1"] += 1
        elif "more realistic image: image 2" in response_lower:
            counts["Image 2"] += 1
        else:
            counts["Unknown/Error"] += 1

    # 如果所有文件已处理完毕，直接打印结果并返回
    if not to_process:
        print("  > 无需处理，跳至下一个。")
        print(f"\n  --- 📊 比较总结 ({os.path.basename(image_dir_1)}) ---")
        print(f"  文件夹1 ({os.path.basename(image_dir_1)}) 更逼真: {counts['Image 1']} 次")
        print(f"  文件夹2 ({os.path.basename(image_dir_2)}) 更逼真: {counts['Image 2']} 次")
        print(f"  无法判断 / 出现错误: {counts['Unknown/Error']} 次")
        return

    # 3. 多线程处理未完成的图像对
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(output_jsonl, "a", encoding="utf-8") as f:
        futures = {executor.submit(process_image_pair, p1, p2): os.path.basename(p1) for p1, p2 in to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Comparing {os.path.basename(image_dir_1)}"):
            img_name = futures[future]
            try:
                result = future.result()
                response_text = result.get("response")

                if response_text:
                    # 使用线程锁确保安全写入文件
                    with write_lock:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

                    # 更新实时统计
                    response_lower = response_text.lower()
                    if "more realistic image: image 1" in response_lower:
                        counts["Image 1"] += 1
                    elif "more realistic image: image 2" in response_lower:
                        counts["Image 2"] += 1
                    else:
                        counts["Unknown/Error"] += 1
                else:
                    counts["Unknown/Error"] += 1
                    print(f"\n[Warning] 图像 {img_name} 处理出错: {result.get('error')}")

            except Exception as e:
                counts["Unknown/Error"] += 1
                print(f"\n[Critical] 任务 {img_name} 失败: {e}")

    print(f"\n  > ✅ 全部完成。 比较结果已保存到 -> {output_jsonl}")

    # 4. 打印最终总结
    print(f"\n  --- 📊 比较总结 ({os.path.basename(image_dir_1)}) ---")
    print(f"  测试组 ({os.path.basename(image_dir_1)}) 更逼真: {counts['Image 1']} 次")
    print(f"  基准组 ({os.path.basename(BENCHMARK_DIR)}) 更逼真: {counts['Image 2']} 次")
    print(f"  无法判断 / 出现错误: {counts['Unknown/Error']} 次")


# ==========================================
# 5. Main Controller (主入口)
# ==========================================

def main():
    """
    主控制器，循环执行配置列表中所有的模型比较任务。
    """
    print("===== 🤖 开始批量图像逼真度评估任务 =====")
    print(f"基准目录 (Image 2): {BENCHMARK_DIR}")
    print(f"结果输出目录: {BASE_OUTPUT_DIR}")
    print(f"总任务数: {len(DIRS_TO_TEST)}")
    print("-----------------------------------------")

    for i, model_dir in enumerate(DIRS_TO_TEST):
        # 尝试从路径中提取模型名称，例如 'gpt', 'gemini'
        # 路径结构预期: .../gpt/photo -> gpt
        try:
            model_name = os.path.basename(os.path.dirname(model_dir))
        except Exception:
            model_name = f"task_{i}"
        
        # 自动生成对应的输出文件名
        output_file_path = os.path.join(BASE_OUTPUT_DIR, f"comparisons_{model_name}.jsonl")

        print(f"\n===== 🚀 [任务 {i+1}/{len(DIRS_TO_TEST)}] 开始比较 =====")
        print(f"  Image 1 (测试组): {model_name}")
        print(f"  Image 2 (基准组): {os.path.basename(BENCHMARK_DIR)}")
        print(f"  输出文件:         {output_file_path}")
        print("-----------------------------------------")

        # 执行单次比较
        run_comparison(
            image_dir_1=model_dir,
            image_dir_2=BENCHMARK_DIR,
            output_jsonl=output_file_path
        )

        print(f"===== ✅ [任务 {i+1}/{len(DIRS_TO_TEST)}] {model_name} 比较完成 =====")

    print("\n===== 🎉 所有批量评估任务均已完成! =====")

if __name__ == "__main__":
    main()