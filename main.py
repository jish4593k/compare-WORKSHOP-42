import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from skimage.metrics import structural_similarity as ssim

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img /= 255.0  # Normalize image
    return img

def compare_images(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    ssim_value = ssim(imageA, imageB, data_range=imageB.max() - imageB.min())
    return mse, ssim_value

def plot_images(imageA, imageB, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(imageA)
    ax1.set_title("Image A")
    ax1.axis('off')
    ax2.imshow(imageB)
    ax2.set_title("Image B")
    ax2.axis('off')
    plt.suptitle(title)
    plt.show()

def batch_image_comparison(original_image_path, image_folder, threshold_mse, threshold_ssim):
    original_image = load_and_preprocess_image(original_image_path)
    similar_images = []
    dissimilar_images = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            compared_image = load_and_preprocess_image(image_path)
            mse, ssim_value = compare_images(original_image, compared_image)

            if mse <= threshold_mse and ssim_value >= threshold_ssim:
                similar_images.append((image_path, mse, ssim_value))
            else:
                dissimilar_images.append((image_path, mse, ssim_value))

    return similar_images, dissimilar_images

def visualize_comparison_results(similar_images, dissimilar_images):
    print("Similar Images:")
    for img_path, mse, ssim_value in similar_images:
        print(f"Image: {img_path}, MSE: {mse:.2f}, SSIM: {ssim_value:.2f}")
        img = load_and_preprocess_image(img_path)
        plt.figure()
        plt.title(f"Similar Image: MSE: {mse:.2f}, SSIM: {ssim_value:.2f}")
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    print("Dissimilar Images:")
    for img_path, mse, ssim_value in dissimilar_images:
        print(f"Image: {img_path}, MSE: {mse:.2f}, SSIM: {ssim_value:.2f}")

if __name__ == "__main__":
    original_image_path = "images/jp_gates_original.png"
    image_folder = "images/modified_images"
    threshold_mse = 0.1  # Adjust the threshold values
    threshold_ssim = 0.7

    similar_images, dissimilar_images = batch_image_comparison(original_image_path, image_folder, threshold_mse, threshold_ssim)

    visualize_comparison_results(similar_images, dissimilar_images)

