import os
from PIL import Image

# 输入和输出文件夹路径
input_folder = r"D:\work\Study\TestImage"  # 替换为你的图片文件夹路径
output_folder = r"D:\work\Study\TestImage\Variant"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片文件
for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)

    # 检查文件是否为图片
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            # 打开图片
            with Image.open(input_path) as img:
                # 检查图片分辨率是否需要降采样
                if img.width > 512 or img.height > 512:
                    # 调整图片大小为 512x512
                    img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)

                    # 保存到输出文件夹
                    output_path = os.path.join(output_folder, file_name)
                    img_resized.save(output_path)

                    print(f"Resized and saved: {output_path}")
                else:
                    print(f"Skipped (already small): {input_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")