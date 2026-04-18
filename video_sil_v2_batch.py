#!/usr/bin/env python3
"""视频人物黑色剪影处理脚本 - v23 批量推理优化版（减少 CPU-GPU 交互）"""
import argparse
import cv2
import numpy as np
import sys
import time
import os
import tempfile
from ultralytics import YOLO
import subprocess

# 解决 PyTorch 2.6+ 权重加载安全问题
import torch
_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load


def _normalize_result_item(item):
    """标准化结果项"""
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return item[0], item[1]
    return item, None


def _extract_segmentation_masks(results, class_filter=None):
    """提取分割 mask（参考正确实现）"""
    masks = []
    if results is None:
        return masks
    
    for r in results:
        det, seg_data = _normalize_result_item(r)
        
        # 获取 mask 数据
        if seg_data is None:
            if not hasattr(det, 'masks') or det.masks is None:
                continue
            seg_data = getattr(det.masks, 'data', None)
        
        try:
            mask_list = list(seg_data) if isinstance(seg_data, (list, tuple)) else list(getattr(seg_data, 'data', []))
        except Exception:
            mask_list = [seg_data]
        
        # 按类别过滤
        indices = None
        if class_filter is not None and hasattr(det, 'boxes') and hasattr(det.boxes, 'cls'):
            try:
                cls_data = det.boxes.cls
                cls_np = cls_data.cpu().numpy() if hasattr(cls_data, 'cpu') else np.array(cls_data)
                cls_np = np.asarray(cls_np).flatten()
                indices = np.where(cls_np == class_filter)[0]
            except Exception:
                indices = None
        
        if indices is not None:
            mask_list = [mask_list[i] for i in indices if i < len(mask_list)]
        
        # 转换为 numpy
        for mask in mask_list:
            if mask is None:
                continue
            try:
                mask_np = mask.cpu().numpy()
            except Exception:
                try:
                    mask_np = np.array(mask)
                except Exception:
                    continue
            if mask_np.size == 0:
                continue
            masks.append(mask_np)
    
    return masks


def _extract_boxes(results, class_filter=0):
    """提取检测框（参考正确实现）"""
    boxes = []
    if results is None:
        return boxes
    
    for r in results:
        det, _ = _normalize_result_item(r)
        
        if not hasattr(det, 'boxes') or det.boxes is None:
            continue
        
        try:
            cls_data = getattr(det.boxes, 'cls', None)
            xyxy_data = getattr(det.boxes, 'xyxy', None)
            
            if xyxy_data is None:
                xyxy_data = getattr(det.boxes, 'xyxyn', None)
            if xyxy_data is None:
                xyxy_data = getattr(det.boxes, 'data', None)
            if xyxy_data is None:
                continue
            
            cls_np = None
            if cls_data is not None:
                try:
                    cls_np = cls_data.cpu().numpy() if hasattr(cls_data, 'cpu') else np.array(cls_data)
                    cls_np = np.asarray(cls_np).flatten()
                except Exception:
                    cls_np = None
            
            xyxy_np = xyxy_data.cpu().numpy() if hasattr(xyxy_data, 'cpu') else np.array(xyxy_data)
            xyxy_np = np.asarray(xyxy_np)
            
            if xyxy_np.ndim == 3 and xyxy_np.shape[0] == 1:
                xyxy_np = xyxy_np[0]
            
            if xyxy_np.ndim == 2:
                if xyxy_np.shape[1] >= 6:
                    coords = xyxy_np[:, :4]
                    cls_from_data = xyxy_np[:, 5]
                elif xyxy_np.shape[1] == 5:
                    coords = xyxy_np[:, :4]
                    cls_from_data = xyxy_np[:, 4]
                elif xyxy_np.shape[1] == 4:
                    coords = xyxy_np[:, :4]
                    cls_from_data = None
                else:
                    continue
                
                if cls_np is None and cls_from_data is not None:
                    cls_np = np.asarray(cls_from_data).flatten()
                
                for idx, box in enumerate(coords):
                    if cls_np is None or idx >= len(cls_np) or cls_np[idx] == class_filter:
                        boxes.append(box.astype(np.int32))
        except Exception:
            continue
    
    return boxes


def _draw_boxes(frame, boxes, color=(0, 0, 255), thickness=1, label='person'):
    """绘制检测框"""
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1] - 1, int(x2)), min(frame.shape[0] - 1, int(y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def process_video(input_path, output_path, use_gpu=True, batch_size=4):
    """处理视频：检测人体并遮挡为黑色剪影 - 批量推理优化版"""
    
    DEVICE = 'cuda:0' if use_gpu else 'cpu'
    model_path = 'yolov8s-seg.pt'
    
    print(f"{'='*50}")
    print(f"视频人物黑色剪影处理 - v23 批量推理优化版")
    print(f"{'='*50}")
    print(f"模型路径：{model_path}")
    print(f"使用设备：{DEVICE}")
    print(f"批处理大小：{batch_size}")

    # 检查 GPU 是否支持 FP16
    device_capability = torch.cuda.get_device_capability()
    if int(device_capability[0]) < 7:
        print(f"GPU 计算能力{device_capability}不支持 FP16，使用 FP32")
        return None
    
    model = YOLO(model_path)
    if use_gpu:
        # model.model.half()  # 转换为 FP16 精度
        model.to(DEVICE)
    
    # 推理参数
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.25
    
    # 设置临时文件路径（使用 H 盘）
    if os.path.exists('H:\\'):
        temp_dir = 'H:\\'
        print(f"  临时目录：H:\\ (内存盘)")
    else:
        temp_dir = tempfile.gettempdir()
        print(f"  临时目录：{temp_dir}")
    
    temp_path = os.path.join(temp_dir, 'video_sil_tmp.jpg')
    print(f"{'='*50}\n")
    
    # 模型预热 - 使用批量预热提高 GPU 利用率
    print("进行模型批量预热...")
    try:
        dummy_images = []
        for _ in range(batch_size):
            dummy_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, dummy_image)
            dummy_images.append(temp_path)
        
        # 批量预热 - 一次处理多帧，提高 GPU 利用率
        _ = model.predict(source=dummy_images, task='segment', conf=CONF_THRESHOLD, 
                         verbose=False, imgsz=IMG_SIZE, batch=batch_size)
        print("✅ 批量预热成功")
    except Exception as e:
        print(f"⚠️ 预热失败：{e}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件：{input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息 - FPS: {fps}, 分辨率：{width}x{height}, 总帧数：{total}")

    if fps <= 0 or fps > 120:
        print(f"⚠️ FPS {fps} 异常，使用默认 25 FPS")
        fps = 25.0

    # 老的 MPEG-4 Part 2 格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"错误：无法创建输出文件：{output_path}")
        cap.release()
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    start_time = time.time()
    frame_num = 0
    total_infer_time = [0.0]  # Use list for mutable reference in nested function
    
    print("开始处理...")
    
    # 帧缓冲区用于批量推理 - 核心优化点！
    frame_buffer = []
    original_frames = []  # 保存原始帧用于后续处理
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # 处理剩余缓冲帧
            if frame_buffer:
                _process_batch(model, frame_buffer, original_frames, width, height, kernel, out,
                              CONF_THRESHOLD, len(frame_buffer), total_infer_time, IMG_SIZE, frame_num)
            break
        
        try:
            # 将帧添加到缓冲区
            frame_buffer.append(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)))  # 预缩放至模型输入尺寸
            original_frames.append(frame.copy())
            
            # 当缓冲区达到批处理大小时进行批量推理
            if len(frame_buffer) >= batch_size:
                _process_batch(model, frame_buffer, original_frames, width, height, kernel, out,
                              CONF_THRESHOLD, batch_size, total_infer_time, IMG_SIZE, frame_num)
                frame_buffer = []
                original_frames = []
        
        except Exception as e:
            import traceback
            print(f"推理失败：{e}")
            traceback.print_exc()
            if original_frames:
                # 移除失败的帧
                original_frames.pop()
                if frame_buffer:
                    frame_buffer.pop()
            out.write(frame)
            frame_num += 1
            continue
        
        frame_num += 1
    
    # 处理最后一批剩余帧（如果视频结束但缓冲区还有帧）
    if frame_buffer:
        _process_batch(model, frame_buffer, original_frames, width, height, kernel, out,
                      CONF_THRESHOLD, len(frame_buffer), total_infer_time)
    
    cap.release()
    
    total_time = time.time() - start_time
    out.release()
    
    print('\n' + '='*50)
    print('=== 性能分析报告 ===')
    print(f'总帧数：{frame_num}')
    print(f'总耗时：{total_time:.2f}秒')
    print(f'平均 FPS: {frame_num/total_time:.1f}')
    
    if frame_num > 0:
        avg_infer_ms = total_infer_time[0] * 1000 / frame_num
        print(f'\n每帧平均耗时:')
        print(f'  YOLO 推理：{avg_infer_ms:.2f}ms')
    
    print(f'\n已保存检测结果到：{output_path}')
    print('='*50)

    # 转码为 H.265 (HEVC) 格式，输出文件名加后缀 '_h265'
    name, ext = os.path.splitext(output_path)
    h265_output = f"{name}_h265.mp4"

    # ffmpeg -i input.mp4 -c:v hevc_nvenc -preset p4 -cq 23 -c:a aac -b:a 128k output.hevc.mp4
    cmd = [
        'ffmpeg',
        '-i', output_path,
        '-c:v', 'hevc_nvenc',      # ← 改为 GPU 编码器
        '-preset', 'p4',
        '-cq', '23',               # ← 替代 -crf
        '-c:a', 'aac',
        '-b:a', '128k',
        '-y',
        h265_output
    ]

    # ffmpeg -i input.mp4 -c:v libx265 -crf 23 -preset medium -c:a aac -b:a 128k output.hevc.mp4 
    # FFmpeg 命令：高质量压缩，保留音频，使用标准封装
    # cmd = [
    #     'ffmpeg',
    #     '-i', output_path,           # 输入文件（你刚保存的检测结果）
    #     '-c:v', 'libx265',           # 使用 H.265 编码器
    #     '-crf', '23',                # 质量平衡值（18-28，23 为默认）
    #     '-preset', 'medium',         # 编码速度与压缩率平衡
    #     '-c:a', 'aac',               # 音频编码
    #     '-b:a', '128k',              # 音频比特率
    #     '-y',                        # 自动覆盖输出文件
    #     h265_output                  # 输出文件
    # ]

    print(f"正在将 {output_path} 转码为 H.265 格式...")
    try:
        # ffmpeg -i input.mp4 -c:v hevc_nvenc -preset p4 -cq 23 -c:a aac -b:a 128k output.hevc.mp4

        # result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        # print(f"✅ 转码成功！H.265 文件已保存至：{h265_output}")

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        output = result.stdout.decode('utf-8', errors='ignore')  # 安全解码，忽略非法字符
        print(output)  # 可选：打印日志
        
        
        original_size = os.path.getsize(output_path) / 1024 / 1024
        new_size = os.path.getsize(h265_output) / 1024 / 1024
        print(f"   原文件大小：{original_size:.1f} MB → 新文件大小：{new_size:.1f} MB（压缩率：{new_size/original_size*100:.0f}%）")
    except subprocess.CalledProcessError as e:
        print("❌ 转码失败！错误信息：")
        print(e.output)
    except FileNotFoundError:
        print("❌ 未找到 FFmpeg，请确保已安装并添加到系统 PATH。")
        print("   下载地址：https://ffmpeg.org/download.html")


def _process_batch(model, frame_buffer, original_frames, width, height, kernel, out, conf_threshold, actual_batch_size, total_infer_time_ref, imgsz=640, start_frame_num=0):
    """批量处理帧 - 核心优化函数
    
    Args:
        model: YOLO 模型
        frame_buffer: 缩放后的帧缓冲区 (imgsz x imgsz)
        original_frames: 原始帧列表
        width: 视频宽度
        height: 视频高度
        kernel: 形态学核
        out: VideoWriter 对象
        conf_threshold: 置信度阈值
        actual_batch_size: 实际批处理大小
        total_infer_time_ref: 推理时间引用 (用于统计)
        imgsz: 模型输入尺寸 (默认 640)
        start_frame_num: 起始帧号 (用于 debug 输出)
    """
    if not frame_buffer or not original_frames:
        return
    
    # 批量推理 - 关键优化点！一次处理多帧，减少 CPU-GPU 交互次数
    infer_start = time.time()
    results = model.predict(
        source=frame_buffer,           # 传入缩放后的图像列表
        task='segment',
        conf=conf_threshold,
        verbose=False,
        save=False,
        show=False,
        imgsz=imgsz,
        batch=actual_batch_size       # 批量推理参数
    )
    total_infer_time_ref[0] += (time.time() - infer_start)
    
    # 批量处理所有帧的结果 - 深度优化：减少 resize 和循环开销
    # 预计算缩放比例，避免重复计算
    scale_x = width / imgsz
    scale_y = height / imgsz
    
    # 调试计数器 - 只在整个视频的前 12 帧输出 debug
    debug_frame_limit = 12
    
    for idx, (original_frame, result) in enumerate(zip(original_frames, results)):
        masks = _extract_segmentation_masks([result], class_filter=0)
        boxes = _extract_boxes([result], class_filter=0)
        
        # 调试输出（仅前 12 帧）- 使用传入的起始帧号计数
        current_frame_num = start_frame_num + idx
        if current_frame_num < debug_frame_limit:
            print(f"DEBUG frame={current_frame_num}: masks_count={len(masks)}, boxes_count={len(boxes)}")
        
        # 处理所有 mask - 优化：直接在缩略图上操作，避免 resize
        for mask_np in masks:
            if not isinstance(mask_np, np.ndarray):
                continue
            
            mask_np = np.squeeze(mask_np)
            if mask_np.ndim != 2:
                continue
            
            # 将 640x640 mask 缩放到原始尺寸并应用
            mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
            mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
            
            # 使用 INTER_AREA 进行缩小，INTER_LINEAR 进行放大更平滑
            resize_mode = cv2.INTER_AREA if scale_x < 1 else cv2.INTER_LINEAR
            mask_resized = cv2.resize(mask_dilated, (width, height), interpolation=resize_mode)
            original_frame[mask_resized > 0] = [0, 0, 0]
        
        # 如果没有 mask，尝试用 box 绘制
        if len(masks) == 0 and len(boxes) > 0:
            _draw_boxes(original_frame, boxes, color=(0, 255, 0), thickness=1, label='person')
        
        # 添加检测文本 - 优化：减少重复计算
        detection_text = 'Find' if (len(masks) > 0 or len(boxes) > 0) else 'NA'
        text_pos = (10, 25)
        text_color = (0, 255, 0) if detection_text == 'Find' else (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 1
        text_size, _ = cv2.getTextSize(detection_text, font, font_scale, thickness)
        cv2.rectangle(original_frame, (text_pos[0] - 4, text_pos[1] - text_size[1] - 4),
                      (text_pos[0] + text_size[0] + 4, text_pos[1] + 4), (0, 0, 0), -1)
        cv2.putText(original_frame, detection_text, text_pos, font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        out.write(original_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='视频人物黑色剪影处理（批量推理优化版）')
    parser.add_argument('input', help='输入视频路径')
    parser.add_argument('-o', '--output', default='output.mp4', help='输出视频路径')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    parser.add_argument('--batch', type=int, default=4, help='批处理大小（默认 4）')
    args = parser.parse_args()
    
    print(f"Python 解释器：{sys.executable}")
    process_video(args.input, args.output, use_gpu=not args.cpu, batch_size=args.batch)
