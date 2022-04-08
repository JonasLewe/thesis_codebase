import os
import json
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from utils.utils import show, get_img_array, get_pil_img, get_img_draw
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array


def generate_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_superimposed_gradcam_img(img, heatmap, cam_path="cam.jpg", alpha=0.8, save=False, display=False):
    # Load the original image
    #img = keras.preprocessing.image.load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)
    
    # Save the superimposed image
    if save:
        superimposed_img.save(cam_path)

    # Display superimposed Grad CAM image
    if display:
        #display(Image(cam_path))
        show(superimposed_img)
    
    return superimposed_img


def cam_display(src_img, superimposed_img, img_name, preds, draw_text, cam_img_output_path):
    text = f'IMG_NAME: {img_name}\nPrediction: {preds}'
    images = [src_img, superimposed_img]
    
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)

    new_img = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for im in images:
        new_img.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_img = ImageOps.expand(new_img, border=50, fill=(255,255,255))
    new_img = new_img.resize(tuple(i*2 for i in new_img.size), resample=Image.BOX)
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.truetype("FONTS/arial.ttf", 40)
    if draw_text:
        draw.text((10,0),text,(0,0,0),font=font)
    if cam_img_output_path:
        new_img.save(os.path.join(cam_img_output_path, img_name))
    show(new_img)
    

def cam_pipeline(BASE_IMG_DIR, img_name, json_img, IMAGE_SIZE, model, last_conv_layer_name, cam_img_output_path, display=False, draw_text=True):
    img_path = os.path.join(BASE_IMG_DIR, img_name)
    img_array = get_img_array(img_path, size=IMAGE_SIZE)
    img = Image.fromarray(img_array)

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # check prediction
    new_image = get_img_array(img_path, expand_dims=True, normalize=True)
    preds = model.predict(new_image)
    #print("Predicted:", decode_predictions(preds, top=1)[0])
    
    # Generate class activation heatmap
    heatmap = generate_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    if display:
        superimposed_img = get_superimposed_gradcam_img(img, heatmap)
        json_img = json_img.resize(IMAGE_SIZE, Image.ANTIALIAS)
        #cam_display(json_img, superimposed_img, img_name, preds, draw_text, cam_img_output_path)
    return heatmap


def get_json_img_name(json_file_name):
    img_name = f"{json_file_name.split('.')[0]}.jpg"
    return img_name


def draw_json_polygons(img_name, json_file_name, class_1_img_folder, polygon_label_folder, image_size):
    img_path = os.path.join(class_1_img_folder, img_name)
    img = get_pil_img(img_path, image_size)
    draw = get_img_draw(img_path, image_size)
    
    # Opening JSON file
    json_path = os.path.join(polygon_label_folder, json_file_name)
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    # iterate over all olivine polygons
    num_of_polygons = len(json_data['shapes'])
    for i in range(num_of_polygons): # iterate over different olivine crystals
        polygon_coordinates = [tuple(x) for x in json_data['shapes'][i]['points']]
        # print(polygon_coordinates)
        polygon_coordinates.append(polygon_coordinates[0]) # polygon needs to be closed
        for j in range(len(polygon_coordinates)-1): # iterate over coordinates of single crystal
            draw.line(polygon_coordinates[j] + polygon_coordinates[j+1], fill='red', width=3)
    return img