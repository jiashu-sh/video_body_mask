
import os
import sys
import multiprocessing
from pathlib import Path
from datetime import datetime

def process_video(input_path, output_path):
    """
    执行单个视频处理命令：python video_sil_v2_batch.py "输入路径" -o "输出路径" --batch 8
    """
    cmd = f'python video_sil_v2_batch.py "{input_path}" -o "{output_path}" --batch 8'
    print(f"[进程 {os.getpid()}] 正在处理: {input_path} → {output_path}")
    result = os.system(cmd)
    if result == 0:
        print(f"[进程 {os.getpid()}] ✅ 成功: {output_path}")
    else:
        print(f"[进程 {os.getpid()}] ❌ 失败: {input_path}")

def main():
    # 检查命令行参数：必须为3个：输入文件夹、输出文件夹、并行数
    if len(sys.argv) != 4:
        print("错误: 参数不足。使用方法: python batch_video_sil.py <输入文件夹> <输出文件夹> <并行数量>")
        print("示例: python batch_video_sil.py \"C:\\\\gitcode\\\\python\\\\video_human_blurring\\\\nvr_export2\" \"C:\\\\gitcode\\\\python\\\\video_human_blurring\\\\output\" 4")
        sys.exit(1)

    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])

    # 打印输出当前时间：
    starttime = datetime.now()
    print(f"开始时间: {starttime.strftime('%Y-%m-%d %H:%M:%S')}")

    # exit(0)  # --- IGNORE --- 这行代码是为了测试参数传递，实际使用时请删除或注释掉
    # ✅ 修复点：正确使用索引提取参数
    input_folder = sys.argv[1]    # 第一个参数：输入文件夹路径
    output_folder = sys.argv[2]   # 第二个参数：输出文件夹路径
    parallel_count = int(sys.argv[3])  # 第三个参数：并行进程数，转为整数

    # 验证输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹不存在 - {input_folder}")
        sys.exit(1)

    # 自动创建输出文件夹（若不存在）
    os.makedirs(output_folder, exist_ok=True)
    print(f"✅ 输出目录已准备: {output_folder}")

    # 获取所有 .mp4 文件,并排序以确保处理顺序一致按文件名排序

    video_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith('.webm') or f.lower().endswith('.mp4')
    ])

    # 取前 N 个文件
    selected_files = video_files[:parallel_count]
    if not selected_files:
        print(f"警告: 在 {input_folder} 中未找到任何 .mp4 文件")
        sys.exit(0)

    print(f"找到 {len(video_files)} 个 .mp4 文件，将处理前 {len(selected_files)} 个：")
    for f in selected_files:
        print(f"  - {f}")

    # 构造输出路径列表：输出文件夹/原文件名-output-N.mp4
    tasks = []
    for idx, filename in enumerate(selected_files, 1):
        input_full = os.path.join(input_folder, filename)
        name_without_ext = Path(filename).stem
        output_filename = f"{name_without_ext}-output-{idx}.mp4"
        output_full = os.path.join(output_folder, output_filename)
        tasks.append((input_full, output_full))

    # 并行执行处理任务
    with multiprocessing.Pool(processes=parallel_count) as pool:
        pool.starmap(process_video, tasks)

    # 打印输出当前时间：
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 计算总耗时：
    print(f"总耗时: {datetime.now() - starttime}")
    print("\n✅ 所有任务已完成。")

if __name__ == "__main__":
    main()
