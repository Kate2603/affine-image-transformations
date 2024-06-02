import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Завантаження зображення
def download_image(image_url, save_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        print(f"Image downloaded and saved to {save_path}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
    return save_path

# Завантаження зображення з інтернету
image_url = 'https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg'
image_path = 'downloaded_image.jpg'
download_image(image_url, image_path)

# Завантаження зображення за допомогою OpenCV
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image at path {image_path}")
else:
    print("Image loaded successfully!")

# Афінні перетворення
def affine_transformations(image):
    rows, cols, ch = image.shape

    # 1. Зменшення в 1/2 рази по осі OX і збільшення в 3 рази по осі OY
    scale_matrix = np.array([[0.5, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=np.float32)
    scaled_image = cv2.warpPerspective(image, scale_matrix, (cols, rows))

    # 2. Відображення відносно початку координат
    reflection_matrix = np.array([[-1, 0, cols], [0, -1, rows], [0, 0, 1]], dtype=np.float32)
    reflected_image = cv2.warpPerspective(image, reflection_matrix, (cols, rows))

    # 3. Перенесення на -3 по осі OX та на 1 по осі OY
    translation_matrix = np.array([[1, 0, -3], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
    translated_image = cv2.warpPerspective(image, translation_matrix, (cols, rows))

    # 4. Зміщення на 60° по осі OY (фактично поворот на 60°)
    angle_60 = np.radians(60)
    cosine_60 = np.cos(angle_60)
    sine_60 = np.sin(angle_60)
    rotation_matrix_60 = np.array([[cosine_60, -sine_60, (1 - cosine_60) * cols / 2 + sine_60 * rows / 2],
                                   [sine_60, cosine_60, (1 - cosine_60) * rows / 2 - sine_60 * cols / 2],
                                   [0, 0, 1]], dtype=np.float32)
    rotated_image_60 = cv2.warpPerspective(image, rotation_matrix_60, (cols, rows))

    # 5. Поворот на 30°
    angle_30 = np.radians(30)
    cosine_30 = np.cos(angle_30)
    sine_30 = np.sin(angle_30)
    rotation_matrix_30 = np.array([[cosine_30, -sine_30, (1 - cosine_30) * cols / 2 + sine_30 * rows / 2],
                                   [sine_30, cosine_30, (1 - cosine_30) * rows / 2 - sine_30 * cols / 2],
                                   [0, 0, 1]], dtype=np.float32)
    rotated_image_30 = cv2.warpPerspective(image, rotation_matrix_30, (cols, rows))

    # 6. Об'єднання всіх перетворень в одну матрицю та застосування її до зображення
    combined_matrix = scale_matrix @ reflection_matrix @ rotation_matrix_60 @ rotation_matrix_30
    combined_image = cv2.warpPerspective(image, combined_matrix, (cols, rows))

    return scaled_image, reflected_image, translated_image, rotated_image_60, rotated_image_30, combined_image

if image is not None:
    scaled_image, reflected_image, translated_image, rotated_image_60, rotated_image_30, combined_image = affine_transformations(image)

    # Відображення результатів
    plt.figure(figsize=(12, 8))

    plt.subplot(231), plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)), plt.title('Scaled Image')
    plt.subplot(232), plt.imshow(cv2.cvtColor(reflected_image, cv2.COLOR_BGR2RGB)), plt.title('Reflected Image')
    plt.subplot(233), plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB)), plt.title('Translated Image')
    plt.subplot(234), plt.imshow(cv2.cvtColor(rotated_image_60, cv2.COLOR_BGR2RGB)), plt.title('Rotated Image 60°')
    plt.subplot(235), plt.imshow(cv2.cvtColor(rotated_image_30, cv2.COLOR_BGR2RGB)), plt.title('Rotated Image 30°')
    plt.subplot(236), plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)), plt.title('Combined Transformation')

    plt.show()