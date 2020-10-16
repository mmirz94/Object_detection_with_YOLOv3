from tensorflow import keras
import numpy as np

def preprocess_image(image_path):
    image = keras.preprocessing.image.load_img(image_path)
    original_img_w, original_img_h = image.size
    # load the image with the required size
    image = keras.preprocessing.image.load_img(image_path, target_size=(416,416))
    # convert to numpy array
    image = keras.preprocessing.image.img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    #expand the dimension of the image so that we have 1 sample ((1,416,416,3)) 
    image=np.expand_dims(image, axis=0)
    print("new image shape is:",image.shape)
    return image,original_img_w,original_img_h