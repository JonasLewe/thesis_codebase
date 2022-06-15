import os
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from utils.img import show, get_img_array, get_pil_img 
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array


def generate_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # try to create single input evaluation model for trained multi input model
    # single_image_input = Input(shape=(224,224,3))
    # x = model.layers[1](single_image_input)
    # for new_layer in model.layers[2:-1]:
    #     x = new_layer(x)
    # output_layer = Dense(1, activation="sigmoid")(x)
    # temp_model = Model(inputs=single_image_input, outputs=output_layer)
    # grad_model = Model([temp_model.inputs], [temp_model.get_layer(last_conv_layer_name).output, temp_model.output])

    # print(f"Grad-CAM Model Summary: {grad_model.summary()}")

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


def generate_gradcam_heatmap_dual_input(img_array_0, img_array_1, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model([img_array_0, img_array_1])
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



def generate_gradcam_heatmap_multi_input(ppl_array_0,
                                         ppl_array_30, 
                                         ppl_array_60,
                                         ppl_array_merged,
                                         xpl_array_0, 
                                         xpl_array_30,
                                         xpl_array_60,
                                         xpl_array_merged,
                                         model,
                                         input_size, 
                                         last_conv_layer_name, 
                                         pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    if input_size == 2:
        img_list = [ppl_array_merged, xpl_array_merged]
    elif input_size == 3:
        img_list = [xpl_array_0, xpl_array_30, xpl_array_60]
    else: # if input_size == 6:
        img_list = [ppl_array_0, ppl_array_30, ppl_array_60, xpl_array_0, xpl_array_30, xpl_array_60]

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_list)
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


def cam_display(src_img, superimposed_img, img_name, preds, draw_text, cam_img_output_path, iou_value=None):
    prediction = '%.4f' % round(preds[0][0], 4)
    text = f'IMG_NAME: {img_name}\nPrediction: {prediction}, IoU: {iou_value}'
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
    

def cam_pipeline(BASE_IMG_DIR, img_name, image_size, model, last_conv_layer_name, draw_text=True ,cam_img_output_path="", segmented_img=None):
    # set fixed image size for cam calculation
    image_size = (224,224)
    img_path = os.path.join(BASE_IMG_DIR, img_name)
    img_array = get_img_array(img_path, image_size=image_size, expand_dims=True)
    img = get_pil_img(img_path, image_size)

    # Remove last layer's softmax
    # model.layers[-1].activation = None

    # check prediction
    new_image = get_img_array(img_path, image_size=image_size, expand_dims=True, normalize=True)
    preds = model.predict(new_image)
    #print(f"Predictions from grad_CAM.py:cam_pipeline: {preds}")
    #print(preds[0][0])
    #print(type(preds))
    #print(type(preds[0][0]))
    #print(f"Rounded predictions: {'%.4f' % round(preds[0][0], 4)}")
    #print("Predicted:", decode_predictions(preds, top=1)[0])

    # Remove last layer's softmax
    model.layers[-1].activation = None
    
    # Generate class activation heatmap
    heatmap = generate_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    if cam_img_output_path:
        superimposed_img = get_superimposed_gradcam_img(img, heatmap)
        segmented_img = segmented_img.resize(image_size, Image.ANTIALIAS)
        cam_display(segmented_img, superimposed_img, img_name, preds, draw_text, cam_img_output_path)
    return heatmap


def cam_pipeline_2_view(BASE_IMG_DIRs, img_name, image_size, model, last_conv_layer_name, draw_text=True ,cam_img_output_path="", segmented_img=None):    
    # set fixed image size for cam calculation
    image_size = (224,224)

    img_path_0 = os.path.join(BASE_IMG_DIRs[0], img_name)
    img_array_0 = get_img_array(img_path_0, image_size=image_size, expand_dims=True)
    # img_0 = get_pil_img(img_path_0, image_size)

    img_path_1 = os.path.join(BASE_IMG_DIRs[1], img_name)
    img_array_1 = get_img_array(img_path_1, image_size=image_size, expand_dims=True)
    img_1 = get_pil_img(img_path_1, image_size)

    # check prediction
    new_image_0 = get_img_array(img_path_0, image_size=image_size, expand_dims=True, normalize=True)
    new_image_1 = get_img_array(img_path_1, image_size=image_size, expand_dims=True, normalize=True)
    preds = model.predict([new_image_0, new_image_1])

    # Remove last layer's softmax
    model.layers[-1].activation = None
    
    # Generate class activation heatmap
    heatmap = generate_gradcam_heatmap_dual_input(img_array_0, img_array_1, model, last_conv_layer_name)
    
    if cam_img_output_path:
        superimposed_img = get_superimposed_gradcam_img(img_1, heatmap)
        segmented_img = segmented_img.resize(image_size, Image.ANTIALIAS)
        cam_display(segmented_img, superimposed_img, img_name, preds, draw_text, cam_img_output_path)
    return heatmap


def cam_pipeline_multi_input(BASE_IMG_DIRs,
                             img_name,
                             image_size,
                             input_size,
                             model, 
                             last_conv_layer_name, 
                             draw_text=True, 
                             cam_img_output_path="", 
                             segmented_img=None):    
    # set fixed image size for cam calculation
    image_size = (224,224)

    img_path_ppl_0 = os.path.join(BASE_IMG_DIRs[0], img_name)
    img_array_ppl_0 = get_img_array(img_path_ppl_0, image_size=image_size, expand_dims=True)
    # img_ppl_0 = get_pil_img(img_path_ppl_0, image_size)

    img_path_ppl_30 = os.path.join(BASE_IMG_DIRs[1], img_name)
    img_array_ppl_30 = get_img_array(img_path_ppl_30, image_size=image_size, expand_dims=True)
    # img_ppl_30 = get_pil_img(img_path_ppl_30, image_size)

    img_path_ppl_60 = os.path.join(BASE_IMG_DIRs[2], img_name)
    img_array_ppl_60 = get_img_array(img_path_ppl_60, image_size=image_size, expand_dims=True)
    # img_ppl_60 = get_pil_img(img_path_ppl_60, image_size)

    img_path_ppl_merged = os.path.join(BASE_IMG_DIRs[3], img_name)
    img_array_ppl_merged = get_img_array(img_path_ppl_merged, image_size=image_size, expand_dims=True)
    # img_ppl_merged = get_pil_img(img_path_ppl_merged, image_size)


    img_path_xpl_0 = os.path.join(BASE_IMG_DIRs[4], img_name)
    img_array_xpl_0 = get_img_array(img_path_xpl_0, image_size=image_size, expand_dims=True)
    # img_xpl_0 = get_pil_img(img_path_xpl_0, image_size)

    img_path_xpl_30 = os.path.join(BASE_IMG_DIRs[5], img_name)
    img_array_xpl_30 = get_img_array(img_path_xpl_30, image_size=image_size, expand_dims=True)
    # img_xpl_30 = get_pil_img(img_path_xpl_30, image_size)

    img_path_xpl_60 = os.path.join(BASE_IMG_DIRs[6], img_name)
    img_array_xpl_60 = get_img_array(img_path_xpl_60, image_size=image_size, expand_dims=True)
    # img_xpl_60 = get_pil_img(img_path_xpl_60, image_size)

    img_path_xpl_merged = os.path.join(BASE_IMG_DIRs[7], img_name)
    img_array_xpl_merged = get_img_array(img_path_xpl_merged, image_size=image_size, expand_dims=True)
    img_xpl_merged = get_pil_img(img_path_xpl_merged, image_size)


    # check predictions
    new_image_ppl_0 = get_img_array(img_path_ppl_0, image_size=image_size, expand_dims=True, normalize=True)
    new_image_ppl_30 = get_img_array(img_path_ppl_30, image_size=image_size, expand_dims=True, normalize=True)
    new_image_ppl_60 = get_img_array(img_path_ppl_60, image_size=image_size, expand_dims=True, normalize=True)
    new_image_ppl_merged = get_img_array(img_path_ppl_merged, image_size=image_size, expand_dims=True, normalize=True)

    new_image_xpl_0 = get_img_array(img_path_xpl_0, image_size=image_size, expand_dims=True, normalize=True)
    new_image_xpl_30 = get_img_array(img_path_xpl_30, image_size=image_size, expand_dims=True, normalize=True)
    new_image_xpl_60 = get_img_array(img_path_xpl_60, image_size=image_size, expand_dims=True, normalize=True)
    new_image_xpl_merged = get_img_array(img_path_xpl_merged, image_size=image_size, expand_dims=True, normalize=True)

    if input_size == 2:
        preds = model.predict([new_image_ppl_merged, new_image_xpl_merged])
    elif input_size == 3:
        preds = model.predict([new_image_xpl_0, new_image_xpl_30, new_image_xpl_60])
    else: # if input_size == 6
        preds = model.predict([new_image_ppl_0, new_image_ppl_30, new_image_ppl_60, new_image_xpl_0, new_image_xpl_30, new_image_xpl_60])

    # Remove last layer's softmax
    model.layers[-1].activation = None
    
    # Generate class activation heatmap
    heatmap = generate_gradcam_heatmap_multi_input(img_array_ppl_0, 
                                                   img_array_ppl_30,
                                                   img_array_ppl_60,
                                                   img_array_ppl_merged,
                                                   img_array_xpl_0,
                                                   img_array_xpl_30,
                                                   img_array_xpl_60,
                                                   img_array_xpl_merged,
                                                   model,
                                                   input_size,
                                                   last_conv_layer_name)
    
    if cam_img_output_path:
        superimposed_img = get_superimposed_gradcam_img(img_xpl_merged, heatmap)
        segmented_img = segmented_img.resize(image_size, Image.ANTIALIAS)
        cam_display(segmented_img, superimposed_img, img_name, preds, draw_text, cam_img_output_path)
    return heatmap