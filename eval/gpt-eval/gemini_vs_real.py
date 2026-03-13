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

# 固定的基准文件夹 (Image 2)
BENCHMARK_DIR = "/banchmark/real-img-banchmark"

# 存放所有JSONL输出的基础目录
BASE_OUTPUT_DIR = ""

# (Image 1) 待测试的文件夹列表
DIRS_TO_TEST = [
    # 加上你的例子
    
]

# ==== 2. API 和线程配置 (保持不变) ====
client = OpenAI(
    base_url="",
    api_key=""
)
write_lock = threading.Lock()
MAX_WORKERS = 200

# ==== 3. 工具函数 (保持不变) ====
def encode_image_to_base64(image_path):
    """将单个图像文件编码为Base64。"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def build_prompt(base64_str_1, base64_str_2):
    """构建包含两张图片的API提示。"""
    prompt_text = """Assume you are an advanced image detection expert with deep knowledge of image analysis and AI-generated content recognition. You are given two images. Please evaluate both images and determine which one is more realistic, using the following criteria:
Unnatural details in human features:
Hands: Examine both images for any unnatural or distorted features in the hands (e.g., extra fingers, awkward positioning, or unnatural proportions).
Faces: Pay close attention to the faces in both images. Look for distorted facial features (e.g., misaligned eyes, overly smooth skin, unrealistic lighting on the face, or unnatural facial expressions).
Repetitive elements or artificial patterns:
Check for background elements that seem repetitive, unnatural, or artificial. These could be areas where the background looks out of place, such as objects that don’t fit naturally or have distorted features (e.g., unnatural patterns, blurry or unrealistic edges).
Lighting: Assess the lighting in each image, especially in relation to the hands, faces, and backgrounds. AI-generated images may exhibit overly smooth or artificial lighting, including excessive shine or unrealistic glossiness, making it appear overly polished or "greasy."
Discrepancies in edges: Evaluate the edges of the hands, faces, and background. In AI-generated images, the edges might appear unnaturally sharp or blurry, and the transitions between objects and the background might not blend naturally.
Overall style and consistency: Assess the overall color tones, contrast, and saturation of each image, especially focusing on the hands, faces, and backgrounds. Overly perfect or unnatural color grading may suggest that an image is AI-generated, particularly if the color scheme doesn’t match the natural lighting or environment.
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
    """加载JSONL记录以进行断点续传。"""
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

def process_image_pair(image_path_1, image_path_2, max_retries=5, backoff_base=1.0):
    """处理一对图像并返回AI的比较结果。"""
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
                model="gemini-2.5-pro",
                messages=messages
            )
            content = response.choices[0].message.content
            response_text = (content or "").strip()

            if response_text:
                return {"image_pair": fname, "response": response_text}
            raise RuntimeError("empty response")

        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                sleep_s = backoff_base * (2 ** (attempt - 1)) + random.random() * 0.5
                time.sleep(sleep_s)
                continue
            return {"image_pair": fname, "response": None, "error": last_err or "Max retries exceeded"}

    return {"image_pair": fname, "response": None, "error": last_err or "Unknown error"}

# ==== 4. 核心比较逻辑 (新函数) ====
def run_comparison(image_dir_1, image_dir_2, output_jsonl):
    """
    运行一次完整的比较流程。
    Args:
        image_dir_1 (str): Image 1 文件夹路径。
        image_dir_2 (str): Image 2 文件夹路径。
        output_jsonl (str): 此次比较的结果 .jsonl 文件路径。
    """
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    records, done_images = load_jsonl_records(output_jsonl)

    # 1. 查找文件
    try:
        all_files_1 = {f for f in os.listdir(image_dir_1) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
        all_files_2 = {f for f in os.listdir(image_dir_2) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    except FileNotFoundError as e:
        print(f"\n[Error] 无法找到文件夹: {e.filename}。跳过此次比较。")
        return

    common_files = sorted(list(all_files_1.intersection(all_files_2)))
    all_pairs = [(os.path.join(image_dir_1, f), os.path.join(image_dir_2, f)) for f in common_files]
    to_process = [(p1, p2) for p1, p2 in all_pairs if os.path.basename(p1) not in done_images]
    
    print(f"  > 找到 {len(all_pairs)} 个同名图像对。")
    print(f"  > ✅ 已完成: {len(done_images)} | 🚀 待处理: {len(to_process)}")

    if not to_process:
        print("  > 无需处理，跳至下一个。")
        # 即使无需处理，也打印已有的统计结果
        counts = {"Image 1": 0, "Image 2": 0, "Unknown/Error": 0}
        for rec in records:
            response_lower = rec.get("response", "").lower()
            if "more realistic image: image 1" in response_lower:
                counts["Image 1"] += 1
            elif "more realistic image: image 2" in response_lower:
                counts["Image 2"] += 1
            else:
                counts["Unknown/Error"] += 1
        
        print(f"\n  --- 📊 比较总结 ({os.path.basename(image_dir_1)}) ---")
        print(f"  文件夹1 ({os.path.basename(image_dir_1)}) 更逼真: {counts['Image 1']} 次")
        print(f"  文件夹2 ({os.path.basename(image_dir_2)}) 更逼G真: {counts['Image 2']} 次")
        print(f"  无法判断 / 出现错误: {counts['Unknown/Error']} 次")
        return

    # 2. 多线程处理
    counts = {"Image 1": 0, "Image 2": 0, "Unknown/Error": 0}
    # (关键) 从已有的记录中恢复计数
    for rec in records:
        response_lower = rec.get("response", "").lower()
        if "more realistic image: image 1" in response_lower:
            counts["Image 1"] += 1
        elif "more realistic image: image 2" in response_lower:
            counts["Image 2"] += 1
        else:
            counts["Unknown/Error"] += 1

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(output_jsonl, "a", encoding="utf-8") as f:
        futures = {executor.submit(process_image_pair, p1, p2): os.path.basename(p1) for p1, p2 in to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Comparing {os.path.basename(image_dir_1)}"):
            img_name = futures[future]
            try:
                result = future.result()
                response_text = result.get("response")

                if response_text:
                    with write_lock:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

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

    # 3. 打印总结
    print(f"\n  --- 📊 比较总结 ({os.path.basename(image_dir_1)}) ---")
    print(f"  文件夹1 ({os.path.basename(image_dir_1)}) 更逼真: {counts['Image 1']} 次")
    print(f"  文件夹2 ({os.path.basename(BENCHMARK_DIR)}) 更逼真: {counts['Image 2']} 次")
    print(f"  无法判断 / 出现错误: {counts['Unknown/Error']} 次")


# ==== 5. 主控制器 (新) ====
def main():
    """
    主控制器，循环执行所有比较任务。
    """
    print("===== 🤖 开始批量比较任务 =====")
    print(f"基准 (Image 2): {BENCHMARK_DIR}")
    print(f"输出目录: {BASE_OUTPUT_DIR}")
    print(f"总任务数: {len(DIRS_TO_TEST)}")
    print("---------------------------------")

    for i, model_dir in enumerate(DIRS_TO_TEST):
        # 从路径中提取模型名称，例如 'gpt', 'echo4o'
        # .../short-img/gpt/photo -> .../short-img/gpt -> gpt
        try:
            model_name = os.path.basename(os.path.dirname(model_dir))
        except Exception:
            # 如果路径很奇怪，就用索引
            model_name = f"task_{i}"
        
        # 自动生成输出文件名
        output_file_path = os.path.join(BASE_OUTPUT_DIR, f"comparisons_{model_name}.jsonl")

        print(f"\n===== 🚀 [任务 {i+1}/{len(DIRS_TO_TEST)}] 开始比较 =====")
        print(f"  Image 1 (Test): {model_name}")
        print(f"  Image 2 (Base): {os.path.basename(BENCHMARK_DIR)}")
        print(f"  输出文件:     {output_file_path}")
        print("---------------------------------")

        # 执行单次比较
        run_comparison(
            image_dir_1=model_dir,
            image_dir_2=BENCHMARK_DIR,
            output_jsonl=output_file_path
        )

        print(f"===== ✅ [任务 {i+1}/{len(DIRS_TO_TEST)}] {model_name} 比较完成 =====")

    print("\n===== 🎉 所有批量任务均已完成! =====")

if __name__ == "__main__":
    main()