#!/usr/bin/env python3
"""视频人物黑色剪影处理脚本 - v23 最终稳定版（参考正确实现）"""
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



def process_video(input_path, output_path, use_gpu=True):
    """处理视频：检测人体并遮挡为黑色剪影"""
    
    DEVICE = 'cuda:0' if use_gpu else 'cpu'
    model_path = 'yolov8s-seg.pt'
    
    print(f"{'='*50}")
    print(f"视频人物黑色剪影处理 - v23 最终稳定版")
    print(f"{'='*50}")
    print(f"模型路径：{model_path}")
    print(f"使用设备：{DEVICE}")

    # 检查GPU是否支持FP16
    device_capability = torch.cuda.get_device_capability()
    if int(device_capability[0]) < 7:
        print(f"GPU计算能力{device_capability}不支持FP16，使用FP32")
        return None
    
    model = YOLO(model_path)
    if use_gpu:
        # model.model.half()  # 转换为FP16精度
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
    
    # 模型预热
    print("进行模型预热...")
    try:
        dummy_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        cv2.imwrite(temp_path, dummy_image)
        _ = model.predict(source=temp_path, task='segment', conf=CONF_THRESHOLD, 
                         verbose=False, imgsz=IMG_SIZE)
        print("预热成功")
    except Exception as e:
        print(f"预热失败：{e}")
    
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

    # 老的 ‌MPEG-4 Part 2‌ 格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) #avc1

    # 改用 H.264 编码，兼容性更好，压缩效率更高
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # ✅ H.264 编码，高压缩、高兼容
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"错误：无法创建输出文件：{output_path}")
        cap.release()
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    start_time = time.time()
    frame_num = 0
    total_infer_time = 0.0
    
    print("开始处理...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        infer_start = time.time()
        
        try:
            # 由于 YOLOv8.0.0 的 predict 方法不支持直接传入 numpy 数组，我们需要先保存到临时文件 ----这里注释掉了，因为升级到了yolo 8.3就可以了
            # cv2.imwrite(temp_path, frame)
            
            
            # results = model.predict(
            #     source=temp_path, 
            #     # half=True,
            #     task='segment', 
            #     conf=CONF_THRESHOLD, 
            #     verbose=False, 
            #     imgsz=IMG_SIZE
            # )
            
            results = model.predict(
                source=frame,
                task='segment',
                conf=CONF_THRESHOLD,
                verbose=False,
                save=False,
                show=False,
                imgsz=IMG_SIZE  # 确保 imgsz 参数正确
            )
            
            total_infer_time += (time.time() - infer_start)
            
            # 【关键】使用正确的提取函数
            masks = _extract_segmentation_masks(results, class_filter=0)
            boxes = _extract_boxes(results, class_filter=0)
            
            # 调试输出（仅前 3 帧）
            if frame_num < 3:
                print(f"DEBUG frame={frame_num}: masks_count={len(masks)}, boxes_count={len(boxes)}")
            
            # 处理所有 mask
            for mask_np in masks:
                if not isinstance(mask_np, np.ndarray):
                    continue
                
                mask_np = np.squeeze(mask_np)
                if mask_np.ndim != 2:
                    continue
                
                mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
                mask_resized = cv2.resize(mask_dilated, (width, height), interpolation=cv2.INTER_NEAREST)
                frame[mask_resized > 0] = [0, 0, 0]
            
            # 如果没有 mask，尝试用 box 绘制
            if len(masks) == 0 and len(boxes) > 0:
                _draw_boxes(frame, boxes, color=(0, 255, 0), thickness=1, label='person')
        
        except Exception as e:
            import traceback
            print(f"推理失败：{e}")
            traceback.print_exc()
            out.write(frame)
            frame_num += 1
            continue
        
        # 添加检测文本
        detection_text = 'Find' if len(masks) > 0 or len(boxes) > 0 else 'NA'
        text_pos = (10, 25)
        text_color = (0, 255, 0) if detection_text == 'Find' else (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 1
        text_size, _ = cv2.getTextSize(detection_text, font, font_scale, thickness)
        cv2.rectangle(frame, (text_pos[0] - 4, text_pos[1] - text_size[1] - 4),
                      (text_pos[0] + text_size[0] + 4, text_pos[1] + 4), (0, 0, 0), -1)
        cv2.putText(frame, detection_text, text_pos, font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        out.write(frame)
        frame_num += 1
    
    cap.release()
    
    total_time = time.time() - start_time
    out.release()
    
    print('\n' + '='*50)
    print('=== 性能分析报告 ===')
    print(f'总帧数：{frame_num}')
    print(f'总耗时：{total_time:.2f}秒')
    print(f'平均 FPS: {frame_num/total_time:.1f}')
    
    if frame_num > 0:
        avg_infer_ms = total_infer_time * 1000 / frame_num
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
    #     '-crf', '23',                # 质量平衡值（18-28，23为默认）
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='视频人物黑色剪影处理')
    parser.add_argument('input', help='输入视频路径')
    parser.add_argument('-o', '--output', default='output.mp4', help='输出视频路径')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    args = parser.parse_args()
    
    print(f"Python 解释器：{sys.executable}")
    process_video(args.input, args.output, use_gpu=not args.cpu)
