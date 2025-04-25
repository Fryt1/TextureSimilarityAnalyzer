import os
import random
from PIL import Image, ImageEnhance
import numpy as np

def generate_variants(image, num_variants=30):
    # 确保图片是 RGB 模式
    if image.mode != "RGB":
        print(f"Converting image from {image.mode} to RGB")
        image = image.convert("RGB")  # 将灰度图或其他模式转换为 RGB

    variants = []
    for _ in range(num_variants):
        variant = image.copy()

        # 随机选择一种处理方式
        choice = random.choice(["brightness", "contrast", "resize"])
        
        if choice == "brightness":
            enhancer = ImageEnhance.Brightness(variant)
            variant = enhancer.enhance(random.uniform(0.6, 1.5))

        elif choice == "contrast":
            enhancer = ImageEnhance.Contrast(variant)
            variant = enhancer.enhance(random.uniform(0.7, 1.3))

        elif choice == "resize":
            scale_factor = random.uniform(0.8, 1.3 )
            new_size = (int(variant.width * scale_factor), int(variant.height * scale_factor))
            variant = variant.resize(new_size, Image.Resampling.LANCZOS)

        # 添加随机噪声
        if random.random() < 0.5:
            variant = add_noise(variant, intensity=20)

        variants.append(variant)
    return variants

def add_noise(image, intensity=20):
    # 将图片转换为 numpy 数组
    np_image = np.array(image)

    # 检查图像的通道数
    if len(np_image.shape) == 2:  # 灰度图
        noise = np.random.randint(-intensity, intensity, np_image.shape, dtype='int16')
        np_image = np.clip(np_image + noise, 0, 255)  # 防止溢出
    elif len(np_image.shape) == 3:  # RGB 图像
        noise = np.random.randint(-intensity, intensity, np_image.shape, dtype='int16')
        np_image = np.clip(np_image + noise, 0, 255)  # 防止溢出

    return Image.fromarray(np_image.astype('uint8'))

def process_images(input_folder, output_folder, num_images=10, num_variants=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载所有图片
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if len(image_files) < num_images:
        raise ValueError("图片数量不足，至少需要 {} 张图片".format(num_images))

    # 随机选择至少 num_images 张图片
    selected_images = random.sample(image_files, num_images)

    for image_file in selected_images:
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)

        # 生成变体
        variants = generate_variants(image, num_variants)

        # 保存变体
        base_name, ext = os.path.splitext(image_file)
        for i, variant in enumerate(variants):
            variant.save(os.path.join(output_folder, f"{base_name}_variant_{i+1}{ext}"))

# 示例调用
input_folder = "D:\work\Study\TestImage"  # 输入图片文件夹路径
output_folder = "D:\work\Study\TestImage\Variant"  # 输出变体文件夹路径
process_images(input_folder, output_folder, num_images=30, num_variants=3)