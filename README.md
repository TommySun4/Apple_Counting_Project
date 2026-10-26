# Apple Counting | 苹果自动计数项目 🍎

基于 MinneApple 果园图像数据集，使用 YOLO 训练单类别苹果检测器，实现苹果定位、图像计数、批量导出和误差评估。

## 1. 环境

推荐 Python 3.10-3.12；GPU 训练更快。建议在独立虚拟环境执行：

```bash
python -m pip install -r requirements.txt
```

## 2. 获取公开数据

官方数据集：https://conservancy.umn.edu/ （搜索 MinneApple）
原始代码：https://github.com/nicolaihaeni/MinneApple
整理版：https://github.com/ai-agriculture-circuits-and-systems/apple_minnesota

本项目的 `prepare_dataset.py` **需要 COCO JSON 格式的检测标注**（包含 images、annotations、categories 三个字段）和配套图片。
如果下载的是原始 PNG 实例掩码，需要先使用官方/整理版提供的转换工具生成 COCO 检测标注，不能直接将 PNG 掩码作为 `--coco` 参数。
也可使用已经转换好的 MinneApple COCO 格式标注。
注意：数据集可能包含不同的训练/验证/测试拆分；本项目默认按图片随机 70/15/15 划分。
若需要严格复现论文结果，请使用官方固定划分。不要将同一图片放入不同集合。

假设图片放在 `raw/images/`，COCO 检测标注位于 `raw/instances.json`：

```bash
python prepare_dataset.py --coco raw/instances.json --images raw/images --out dataset
```

## 3. 训练

```bash
python train.py --data dataset/data.yaml --model yolo11n.pt --epochs 80 --imgsz 960 --batch 8
```

输出权重：`runs/apple_detector/weights/best.pt`。
无 GPU 时可增加 `--device cpu --batch 2 --imgsz 640`（速度会明显降低）。

## 4. 自动检测与计数

```bash
python predict.py --weights runs/apple_detector/weights/best.pt --source raw/images --output outputs
```

光照偏暗时可尝试增加 `--clahe`，该参数非必需，需比较验证结果后选择。

输出：
- `outputs/` 内保存带框及苹果数量的图片
- `outputs/counts.csv` 保存每张图的数量

## 5. 评估

```bash
python evaluate.py --weights runs/apple_detector/weights/best.pt --data dataset/data.yaml --dataset dataset
```

输出 mAP50、mAP50-95、MAE、RMSE、`outputs/test_errors.csv`。

计数 MAE = mean(abs(predicted - ground_truth))，越低越好。
检测 mAP 是定位效果指标，与苹果数量误差不同，不能互相替代。

## 6. 针对难点的设计

- **光照**：训练中的 HSV 颜色/亮度抖动；推理可选 CLAHE。
- **遮挡**：果园真实样本 + Mosaic 增强，提高部分遮挡下的鲁棒性。
- **尺度变化**：960 像素输入、多尺度几何扰动；密集小果仍可能漏检。
- **复杂背景**：利用单类监督的检测模型区分树叶、树干与苹果。
- **重叠目标**：使用 YOLO 的后处理/NMS，控制重复计数。

## 7. 限制与注意事项

1. 默认计数的是**照片中可检测的可见苹果**，而不是整棵树被遮住的总产量。
2. 训练和运行前必须下载数据及对应的合法标注，项目包不包含数据与训练权重。
3. 本项目未在真实 MinneApple 数据上完成训练，因此不提供虚构的精度分数。
4. 数据集和 YOLO 软件各自有使用许可；公开展示或商用前应审查对应许可。
5. 同一果园拍摄的相邻帧高度相关，随机拆分可能高估泛化能力。严谨实验应按拍摄场景/树/时间划分。
