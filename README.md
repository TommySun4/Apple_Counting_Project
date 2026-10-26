# Apple Counting | Automated Apple Detection and Counting 🍎

This project uses the MinneApple orchard image dataset to train a single-class apple detector with YOLO. It supports apple detection, image-based counting, batch result export, and error evaluation.

## 1. Environment Setup

Python 3.10–3.12 is recommended. GPU acceleration is recommended for faster training.

Install the required dependencies in a separate virtual environment:

```bash
python -m pip install -r requirements.txt
```

## 2. Dataset Preparation

**Dataset sources:**

- Official dataset: https://conservancy.umn.edu/ (search for MinneApple)
- Original repository: https://github.com/nicolaihaeni/MinneApple
- Organised dataset repository: https://github.com/ai-agriculture-circuits-and-systems/apple_minnesota

The `prepare_dataset.py` script requires COCO JSON detection annotations containing the `images`, `annotations`, and `categories` fields, together with the corresponding images.

If the downloaded dataset contains original PNG instance masks, these must first be converted into COCO detection annotations using the conversion tools provided by the official or organised dataset repositories. PNG masks cannot be passed directly to the `--coco` argument.

Pre-converted MinneApple COCO annotations can also be used.

**Dataset splitting:**

The project uses a random 70/15/15 split for training, validation, and testing by default. Different versions of the dataset may provide different splits.

For strict reproducibility of published results, use the official fixed split. The same image must not appear in multiple subsets.

Assuming the images are stored in `raw/images/` and the COCO annotations are stored in `raw/instances.json`, run:

```bash
python prepare_dataset.py --coco raw/instances.json --images raw/images --out dataset
```

## 3. Model Training

Train the YOLO model using:

```bash
python train.py --data dataset/data.yaml --model yolo11n.pt --epochs 80 --imgsz 960 --batch 8
```

The trained model weights will be saved to:

`runs/apple_detector/weights/best.pt`

If a GPU is unavailable, CPU training can be used with a smaller batch size and image resolution:

`--device cpu --batch 2 --imgsz 640`

However, CPU training will be significantly slower.

## 4. Automatic Apple Detection and Counting

Run inference using:

```bash
python predict.py --weights runs/apple_detector/weights/best.pt --source raw/images --output outputs
```

For images with poor lighting conditions, the optional `--clahe` argument can be used to apply contrast enhancement. Its effectiveness should be evaluated before use.

**Outputs:**

- `outputs/`: Images containing detected bounding boxes and apple counts.
- `outputs/counts.csv`: The number of detected apples in each image.

## 5. Model Evaluation

Evaluate the trained model using:

```bash
python evaluate.py --weights runs/apple_detector/weights/best.pt --data dataset/data.yaml --dataset dataset
```

The evaluation produces:

- mAP50
- mAP50–95
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- `outputs/test_errors.csv`

The counting MAE is calculated as:

\[
MAE = \operatorname{mean}(|predicted-ground\_truth|)
\]

A lower MAE indicates better counting accuracy.

Detection mAP evaluates object localisation performance, whereas counting error measures differences in the number of detected objects. These metrics are not interchangeable.

## 6. Handling Key Challenges

**Lighting variations:** HSV colour and brightness augmentation are applied during training. Optional CLAHE enhancement can be used during inference.

**Occlusion:** Real orchard images and Mosaic augmentation are used to improve robustness when apples are partially occluded.

**Scale variations:** A 960-pixel input resolution and multi-scale geometric augmentation help the model detect apples of different sizes. However, small apples in densely populated regions may still be missed.

**Complex backgrounds:** The single-class supervised detection model learns to distinguish apples from leaves, branches, and other background objects.

**Overlapping objects:** YOLO post-processing and Non-Maximum Suppression (NMS) help reduce duplicate detections and incorrect counting.

## 7. Limitations and Considerations

1. The system counts **visible and detectable apples in an image**, rather than estimating the total number of apples on a tree, including those hidden from view.
2. The dataset and its valid annotations must be downloaded before training or inference. The project package does not include the dataset or trained model weights.
3. The model has not yet been trained on the actual MinneApple dataset. Therefore, no unverified performance scores are reported.
4. The dataset and YOLO framework have their own licensing requirements. Their licences should be reviewed before public distribution or commercial use.
5. Consecutive frames captured in the same orchard can be highly correlated. Random dataset splitting may overestimate generalisation performance. For more rigorous evaluation, images should be separated according to recording scene, tree, or capture time.
