import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# URL зображення
image_url = 'https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg'

# Завантаження зображення
response = requests.get(image_url)
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    # Збереження зображення на локальний диск
    image_path = 'downloaded_image.jpg'
    image.save(image_path)
    print(f"Image downloaded and saved to {image_path}")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

# Завантаження зображення за допомогою OpenCV
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image at path {image_path}")
else:
    print("Image loaded successfully!")

# Перетворення для афінних перетворень
if image is not None:
    rows, cols, ch = image.shape

    # 1. Зменшення в 2 рази по осі OX і збільшення в 3 рази по осі OY
    scale_matrix = np.array([[0.5, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=np.float32)
    scaled_image = cv2.warpPerspective(image, scale_matrix, (cols, rows))

    # 2. Відображення відносно початку координат
    reflection_matrix = np.array([[-1, 0, cols], [0, -1, rows], [0, 0, 1]], dtype=np.float32)
    reflected_image = cv2.warpPerspective(image, reflection_matrix, (cols, rows))

    # 3. Перенесення на -3 по осі OX та на 1 по осі OY
    translation_matrix = np.array([[1, 0, -3], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
    translated_image = cv2.warpPerspective(image, translation_matrix, (cols, rows))

    # 4. Зміщення на 60° по осі OY (фактично поворот на 60°)
    angle = np.radians(60)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation_matrix_60 = np.array([[cosine, -sine, (1 - cosine) * cols / 2 + sine * rows / 2],
                                   [sine, cosine, (1 - cosine) * rows / 2 - sine * cols / 2],
                                   [0, 0, 1]], dtype=np.float32)
    rotated_image_60 = cv2.warpPerspective(image, rotation_matrix_60, (cols, rows))

    # 5. Поворот на 30°
    angle = np.radians(30)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation_matrix_30 = np.array([[cosine, -sine, (1 - cosine) * cols / 2 + sine * rows / 2],
                                   [sine, cosine, (1 - cosine) * rows / 2 - sine * cols / 2],
                                   [0, 0, 1]], dtype=np.float32)
    rotated_image_30 = cv2.warpPerspective(image, rotation_matrix_30, (cols, rows))

    # 6. Об'єднання всіх перетворень в одну матрицю та застосування її до зображення
    combined_matrix = scale_matrix @ reflection_matrix @ rotation_matrix_60 @ rotation_matrix_30
    combined_image = cv2.warpPerspective(image, combined_matrix, (cols, rows))

    # Відображення результатів
    plt.figure(figsize=(12, 8))

    plt.subplot(231), plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)), plt.title('Scaled Image')
    plt.subplot(232), plt.imshow(cv2.cvtColor(reflected_image, cv2.COLOR_BGR2RGB)), plt.title('Reflected Image')
    plt.subplot(233), plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB)), plt.title('Translated Image')
    plt.subplot(234), plt.imshow(cv2.cvtColor(rotated_image_60, cv2.COLOR_BGR2RGB)), plt.title('Rotated Image 60°')
    plt.subplot(235), plt.imshow(cv2.cvtColor(rotated_image_30, cv2.COLOR_BGR2RGB)), plt.title('Rotated Image 30°')
    plt.subplot(236), plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)), plt.title('Combined Transformation')

    plt.show()

# Векторні перетворення
# Вихідний вектор
x = np.array([[2], [1]])

# 1. Зменшити вектор x в 1/2 рази по вісі OX та збільшити в 3 рази по вісі OY
scaling_matrix = np.array([
    [0.5, 0],
    [0, 3]
])
scaled_x = np.dot(scaling_matrix, x)
print("Scaled vector:")
print(scaled_x)

# 2. Відобразити вектор x відносно початку координат
reflection_matrix = np.array([
    [-1, 0],
    [0, -1]
])
reflected_x = np.dot(reflection_matrix, x)
print("Reflected vector:")
print(reflected_x)

# 3. Перенесення на -3 по осі OX та на 1 по осі OY
x_homogeneous = np.array([[2], [1], [1]])  # Додавання однорідної координати
translation_matrix = np.array([
    [1, 0, -3],
    [0, 1,  1],
    [0, 0,  1]
], dtype=np.float32)
translated_x = np.dot(translation_matrix, x_homogeneous)
print("Translated vector:")
print(translated_x[:2])  # Відкидання однорідної координати

# 4. Зміщення вектор x на 60° по вісі OY
shear_matrix_60 = np.array([
    [1, 0],
    [np.tan(np.radians(60)), 1]
])
sheared_x = np.dot(shear_matrix_60, x)
print("Sheared vector by 60° along OY:")
print(sheared_x)

# 5. Поворот вектора x на 30°
theta = np.radians(30)
rotation_matrix_30 = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
rotated_x_30 = np.dot(rotation_matrix_30, x)
print("Rotated vector by 30°:")
print(rotated_x_30)

# 6. Об'єднання перетворень з кроків 1, 2, 4, 5 в одну матрицю та застосування її до вектора x.
combined_matrix = scaling_matrix @ reflection_matrix @ shear_matrix_60 @ rotation_matrix_30
transformed_vector = np.dot(combined_matrix, x)
print("Transformed vector with combined transformations:")
print(transformed_vector)