import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from scipy.ndimage import label as nd_label  # Rename to avoid conflict with tkinter.Label
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# 定义改进后的 ResNet2D 模型
class ResNet2D(nn.Module):
    def __init__(self, num_classes_malignancy):
        super(ResNet2D, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 使用ResNet-18以减少计算量
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为3
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),  # 减少全连接层的大小
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_malignancy)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

# 实例化改进后的模型
pytorch_model_path = "F:\\LUNA16\\resnet18_model.pth"  # 替换为你的 PyTorch 模型路径
model = ResNet2D(num_classes_malignancy=2)
model.load_state_dict(torch.load(pytorch_model_path))
model.eval()

# 加载训练好的 U-net 模型
unet_model_path = "F:\\LUNA16\\UNet_best_Model_checkpoint.h5"
unet_model = load_model(unet_model_path)

# 定义 GUI 应用程序
class LungSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Segmentation and Classification")
        self.root.geometry("1200x800")  # 初始化窗口大小

        self.label = tk.Label(root, text="Select an image file:")
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(side="left", padx=10)

        self.roi_label = tk.Label(self.image_frame)
        self.roi_label.pack(side="right", padx=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))  # 放大字体
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((512, 512))  # Resize to 512x512
        image_array = np.array(image).astype(np.float32)  # Convert to float32

        # Apply the same preprocessing as used during training
        image_array = (image_array - 127.0) / 127.0
        image_array = np.reshape(image_array, (1, 512, 512, 1))  # Add batch dimension

        # Predict the segmentation mask
        predicted_mask = unet_model.predict(image_array)

        # Post-process the predicted mask
        predicted_mask = np.squeeze(predicted_mask)  # Remove batch dimension
        predicted_mask = (predicted_mask >= 0.04).astype(np.uint8)  # Binarize the mask

        labeled_array, num_features = nd_label(predicted_mask)  # Use renamed label function from scipy.ndimage

        if num_features == 0:
            self.display_no_nodule_detected(image)
            return

        target_size = (512, 512)  # 调整大小为与原图相同
        for region in range(1, num_features + 1):
            region_mask = (labeled_array == region)
            if np.sum(region_mask) < 5:  # 设定一个阈值，例如100个像素
                continue

            y_indices, x_indices = np.nonzero(region_mask)
            x1, x2 = x_indices.min() - 30, x_indices.max() + 30
            y1, y2 = y_indices.min() - 30, y_indices.max() + 30

            # 提取原始图像中的对应区域
            roi = np.squeeze(image_array)[y1:y2, x1:x2]
            resized_roi = cv2.resize(roi, target_size)

            # 将 resized_roi 转换为三通道图像以适应 ResNet-18 模型输入要求，并进行归一化处理以确保显示正确的图像
            resized_roi_3ch = cv2.cvtColor((resized_roi * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # 预处理 ROI 以适应 PyTorch 模型
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            roi_tensor = transform(resized_roi_3ch).unsqueeze(0)  # 添加批次维度

            # 使用 ResNet-18 模型进行预测
            with torch.no_grad():
                output = model(roi_tensor)
                _, predicted_class = torch.max(output, 1)

            # 将预测结果映射为良性或恶性
            class_names = ["Malignant", "Benign"]
            prediction_result = class_names[predicted_class.item()]

            self.display_results(image, resized_roi_3ch, prediction_result)
            return

    def display_no_nodule_detected(self, image):
        # Display the original image with a message indicating no nodule detected
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

        # Clear the ROI label and display the result message
        self.roi_label.config(image='')
        self.result_label.config(text="No severe lung nodule detected")

    def display_results(self, image, resized_roi_3ch, prediction_result):
        # Display the original image
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

        # Display the ROI image with the same size as the original image without inverting colors.
        roi_image_pil = Image.fromarray(resized_roi_3ch.astype(np.uint8))
        roi_image_tk = ImageTk.PhotoImage(roi_image_pil.resize((512, 512)))
        self.roi_label.config(image=roi_image_tk)
        self.roi_label.image = roi_image_tk

        # Display the result
        self.result_label.config(text=f"Predicted class for the ROI: {prediction_result}")

# 创建并运行 GUI 应用程序
root = tk.Tk()
app = LungSegmentationApp(root)
root.mainloop()