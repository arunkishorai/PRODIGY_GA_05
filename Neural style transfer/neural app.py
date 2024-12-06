import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

style_transfer_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def preprocess_image(image_path):
    image_data = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_data, channels=3)  
    image = tf.image.convert_image_dtype(image, tf.float32) 
    image = tf.image.resize(image, (256, 256)) 
    image = image[tf.newaxis, :] 
    return image

main_image = preprocess_image('cr7.jpg')  
artistic_image = preprocess_image('famous.jpg') 

plt.imshow(np.squeeze(artistic_image))
plt.axis('off')
plt.title("Style Image")
plt.show()

generated_image = style_transfer_model(tf.constant(main_image), tf.constant(artistic_image))[0]

final_output = np.squeeze(generated_image.numpy()) 
final_output = (final_output * 255).astype(np.uint8)  
output_filename = 'stylized_output.jpg'
cv2.imwrite(output_filename, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))

plt.imshow(final_output)
plt.axis('off')
plt.title("Stylized Image")
plt.show()
