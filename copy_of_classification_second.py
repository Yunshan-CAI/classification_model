# -*- coding: utf-8 -*-
"""Copy of Classification_second.ipynb


Original file is located at
    https://colab.research.google.com/drive/1xcLMt9aSZOcNH14Q9WZz7gXnbqR7_gMk

# **This is to train a simple nail biting classification model based on internet and "11k Hands" data with two categories.**

# Step1: dataloading and preprocessing
"""

from google.colab import drive
drive.mount('/content/drive')

from fastai.vision.all import *
from pathlib import Path
path = Path('/content/drive/MyDrive/nail_data/classification_model_2')

dls = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='valid',
    item_tfms=[Resize(192, method='squish')]
)

"""# Step2: Check the data"""

dls.show_batch(max_n=6, nrows=2)

"""# Step3: Train the model"""

learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(10, base_lr=1e-3)

"""# Step4: Save the model"""

learn.export('/content/drive/MyDrive/nail_data/trained_models/nail_bitten_classifier_2.pkl')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

"""# Step 6: Experimenting with different presizing techniques"""

#改进版本1：基础presizing
dls = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='valid',
    item_tfms=[Resize(460)],  # 先resize到大尺寸，默认crop
    batch_tfms=aug_transforms(size=192, min_scale=0.75)  # GPU上数据增强+最终resize
)

learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(10, base_lr=1e-3)

#改进版本2：针对指甲优化

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,  # 从文件夹名获取标签
    splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(
        size=192,
        min_scale=0.8,
        max_rotate=10,
        max_zoom=1.1,
        max_warp=0.1,
        flip_vert=False
    )
)

dls = dblock.dataloaders(path, bs=16)

learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(10, base_lr=1e-3)

"""# Step 7: Experimenting with different learning rate"""

# 1. 创建learner（但不要训练）
learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])

# 2. 运行lr_find
learn.lr_find()

#Valley Point（FastAI推荐）
learn.fine_tune(10, base_lr=5.75e-4)

#稍微激进一点 (最陡下降区域左侧)
learn.fine_tune(10, base_lr=1e-4)

learn.fine_tune(10, base_lr=2e-3)

learn.fine_tune(10, base_lr=3e-3)

learn.fine_tune(10, base_lr=5e-3)

learn.fine_tune(10, base_lr=7e-3)

learn.fine_tune(10, base_lr=7e-3)

"""# Step 8: Experimenting with different discriminative learning rates"""

#标准Discriminative LR
learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])

# 第一阶段：冻结预训练层，只训练分类头
learn.fit_one_cycle(3, 3e-3)

# 第二阶段：解冻并使用discriminative learning rates
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6, 1e-4))

# 更激进的范围（基于你的lr_find结果）
learn.fit_one_cycle(3, 3e-3)

learn.unfreeze()
# 使用更大的学习率范围
learn.fit_one_cycle(12, lr_max=slice(5e-5, 3e-3))

#三层不同学习率
learn.fit_one_cycle(3, 3e-3)

learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6, 1e-5, 1e-4))

"""# Step 9: Experimenting with different architecture"""

# resnet50
learn = vision_learner(dls, resnet50, metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(10, base_lr=1e-3)

# EfficientNet-B0 - 效率很高的现代架构
learn = vision_learner(dls, efficientnet_b0, metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(10, base_lr=1e-3)

"""# Step 10: See model performance with best presizing, learning rate and architecture"""

dls = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='valid',
    item_tfms=[Resize(460)],  # 先resize到大尺寸，默认crop
    batch_tfms=aug_transforms(size=192, min_scale=0.75)  # GPU上数据增强+最终resize
)

learn = vision_learner(dls, efficientnet_b0, metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(15, base_lr=3e-3)

learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(15, base_lr=3e-3)

learn = vision_learner(dls, efficientnet_b0, metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(15, base_lr=7e-3)

"""# Step 11: Experimenting with progressive resizing"""

# 创建获取DataLoaders的函数
def get_dls(size, bs=16):
    return ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=[Resize(size, method='squish')],
        bs=bs
    )

# Stage 1: 小尺寸快速训练
print("Stage 1: 小尺寸训练 (128x128)")
dls = get_dls(128, bs=32)  # 小图像可以用更大的batch size
learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])
learn.fit_one_cycle(4, 3e-3)  # 用你找到的最佳学习率

# Stage 2: 大尺寸精细训练
print("Stage 2: 大尺寸训练 (224x224)")
learn.dls = get_dls(224, bs=16)  # 大图像用小一点的batch size
learn.fine_tune(6, 1e-3)  # 更低的学习率精细调整

# 使用你发现的最佳参数组合
def get_dls_optimized(size, bs=16):
    return ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=[Resize(460 if size > 192 else size)],  # 大尺寸用presizing
        batch_tfms=aug_transforms(size=size, min_scale=0.75) if size > 192 else None,
        bs=bs
    )

# Stage 1: 小尺寸 + 你的最佳学习率
print("Stage 1: 小尺寸训练 (128x128)")
dls = get_dls_optimized(128, bs=32)
learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])
learn.fit_one_cycle(4, 3e-3)  # 你的最佳学习率

# Stage 2: 大尺寸 + 最佳presizing + 最佳学习率
print("Stage 2: 大尺寸训练 (192x192)")
learn.dls = get_dls_optimized(192, bs=16)
learn.fine_tune(6, 3e-3)  # 继续用你的最佳学习率

# 创建获取DataLoaders的函数
def get_dls(size, bs=16):
    return ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=[Resize(size, method='squish')],
        bs=bs
    )

# 四阶段渐进式训练参数
stages = [
    (96, 32, 3, 3e-3),   # (size, batch_size, epochs, lr)
    (128, 24, 4, 3e-3),
    (192, 16, 4, 2e-3),
    (224, 16, 6, 1e-3)   # 最后用更低学习率精细调整
]

print("🚀 开始四阶段Progressive Resizing训练")
print("="*60)

import gc

learn = None
for i, (size, bs, epochs, lr) in enumerate(stages):
    print(f"\n📍 Stage {i+1}/4: {size}x{size}, batch_size={bs}, epochs={epochs}, lr={lr}")
    print("-" * 40)

    if learn is None:
        # 第一阶段：创建新的learner
        print("✨ 创建新的learner...")
        dls = get_dls(size, bs)
        learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])
        print(f"🎯 开始Stage {i+1}训练...")
        learn.fit_one_cycle(epochs, lr)
    else:
        # 后续阶段：替换DataLoaders并继续训练
        print("🔄 更新DataLoaders...")

        # 清理GPU内存
        torch.cuda.empty_cache()
        gc.collect()

        # 更新DataLoaders
        learn.dls = get_dls(size, bs)

        print(f"🎯 开始Stage {i+1}训练...")
        learn.fit_one_cycle(epochs, lr)

    # 显示当前阶段结果
    print(f"✅ Stage {i+1} 完成!")

print("\n" + "="*60)
print("🏆 四阶段Progressive Resizing训练完成!")
print("="*60)

# 显示最终结果（可选）
print("\n📊 最终验证结果:")
learn.validate()

"""# Step 12: Experimenting with test time agumentation"""

learn.fine_tune(10, base_lr=1e-3)

print("\n🔍 开始TTA测试...")

# 普通验证结果
normal_results = learn.validate()
normal_acc = float(normal_results[2])
normal_precision = float(normal_results[0])
normal_recall = float(normal_results[1])
normal_f1 = float(normal_results[3])

# TTA增强预测
preds_tta, targs = learn.tta()
tta_acc = accuracy(preds_tta, targs).item()

# 计算TTA的其他指标
preds_class = preds_tta.argmax(dim=1)
from sklearn.metrics import precision_score, recall_score, f1_score
tta_precision = precision_score(targs, preds_class, average='weighted')
tta_recall = recall_score(targs, preds_class, average='weighted')
tta_f1 = f1_score(targs, preds_class, average='weighted')

# 结果对比
print(f"\n📊 结果对比:")
print(f"普通验证 - 准确率: {normal_acc*100:.2f}%, F1: {normal_f1*100:.2f}%")
print(f"TTA增强 - 准确率: {tta_acc*100:.2f}%, F1: {tta_f1*100:.2f}%")
print(f"🚀 TTA提升: 准确率 {(tta_acc-normal_acc)*100:+.2f}%, F1 {(tta_f1-normal_f1)*100:+.2f}%")

"""# Step 13: Experimenting with label smoothing"""

import timm
model = timm.create_model('resnet34', pretrained=True, num_classes=dls.c)
learn = Learner(dls, model,
                loss_func=LabelSmoothingCrossEntropy(),
                metrics=[Precision(),Recall(),accuracy,F1Score()])

learn.fine_tune(10, base_lr=1e-3)

"""# Step 14: Optimize weight_decay parameters"""

# 实验不同的weight_decay值
weight_decays = [0.01, 0.1, 0.3]  # 默认是0.01

results = []
for wd in weight_decays:
    print(f"\n🧪 测试Weight Decay = {wd}")

    learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])
    learn.fine_tune(10, base_lr=1e-3, wd=wd)

    # 记录结果
    val_results = learn.validate()
    accuracy_score = float(val_results[2]) * 100
    f1_score = float(val_results[3]) * 100

    results.append({
        'weight_decay': wd,
        'accuracy': accuracy_score,
        'f1': f1_score
    })

    print(f"准确率: {accuracy_score:.2f}%, F1: {f1_score:.2f}%")

# 结果对比
print(f"\n📊 Weight Decay对比结果:")
for result in results:
    print(f"WD={result['weight_decay']}: Acc={result['accuracy']:.2f}%, F1={result['f1']:.2f}%")

"""# Save a good performance model and test (progressive resizing)--not good enough"""

# 创建获取DataLoaders的函数
def get_dls(size, bs=16):
    return ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=[Resize(size, method='squish')],
        bs=bs
    )

# Stage 1: 小尺寸快速训练
print("Stage 1: 小尺寸训练 (128x128)")
dls = get_dls(128, bs=32)  # 小图像可以用更大的batch size
learn = vision_learner(dls, resnet34, metrics=[Precision(),Recall(),accuracy,F1Score()])
learn.fit_one_cycle(4, 3e-3)  # 用你找到的最佳学习率

# Stage 2: 大尺寸精细训练
print("Stage 2: 大尺寸训练 (224x224)")
learn.dls = get_dls(224, bs=16)  # 大图像用小一点的batch size
learn.fine_tune(6, 1e-3)  # 更低的学习率精细调整

learn.export('/content/drive/MyDrive/nail_data/trained_models/nail_bitten_classifier_2_good_performance.pkl')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

"""#Test the model"""

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

model_path = '/content/drive/MyDrive/nail_data/trained_models/nail_bitten_classifier_2.pkl'
learn = load_learner(model_path)

img_path = '/content/drive/MyDrive/nail_data/test_data/huien.JPG'
img = PILImage.create(img_path)

pred_class, pred_idx, probs = learn.predict(img)

confidence = probs[pred_idx].item() * 100

title = f"Prediction result: {pred_class} (Confidence coefficient: {confidence:.1f}%)"
print(title)

import json

# 读取并清理notebook
def clean_notebook_metadata():
    # 假设你的notebook文件名是这个
    notebook_name = "Copy of Classification_second.ipynb"

    try:
        with open(notebook_name, 'r') as f:
            notebook = json.load(f)

        # 清理widgets metadata
        if 'metadata' in notebook:
            if 'widgets' in notebook['metadata']:
                # 选项1: 完全删除widgets metadata
                del notebook['metadata']['widgets']
                print("✅ 删除了widgets metadata")

        # 保存清理后的notebook
        with open(notebook_name.replace('.ipynb', '_clean.ipynb'), 'w') as f:
            json.dump(notebook, f, indent=2)

        print("✅ 清理完成！使用 _clean.ipynb 文件")

    except Exception as e:
        print(f"❌ 错误: {e}")

clean_notebook_metadata()

import os
import json

# 查看当前目录下的所有文件
print("📁 当前目录下的所有文件：")
for file in os.listdir('.'):
    if file.endswith('.ipynb'):
        print(f"✅ Notebook文件：{file}")
    else:
        print(f"📄 其他文件：{file}")
