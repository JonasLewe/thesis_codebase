import imutils
from PIL import Image, ImageDraw


def convert_to_pil(cv2_img):
    return Image.fromarray(cv2_img)


def rotate_img_imutils(cv2_img, degrees):
    return convert_to_pil(imutils.rotate_bound(cv2_img, -degrees))


def rotate_and_mask(cv2_image, degrees): # cv2 image as input
    circle_radius = cv2_image.shape[0]//2 - 5
    
    img = rotate_img_imutils(cv2_image, degrees) # outputs pil image
    
    def get_final_img(img):
        img_radius = img.size[0]//2

        upper_left = img_radius - circle_radius
        upper_left_tuple = (upper_left*3, upper_left*3)
        lower_right = img_radius + circle_radius
        lower_right_tuple = (lower_right*3, lower_right*3)

        # apply mask to img
        bigsize = (img.size[0] * 3, img.size[1] * 3)
        mask = Image.new('L', bigsize, 0)
        draw = ImageDraw.Draw(mask) 
        draw.ellipse(upper_left_tuple + lower_right_tuple, fill=255)
        mask = mask.resize(img.size, Image.ANTIALIAS)
        img.putalpha(mask)

        # convert alpha channel to white
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask = img.split()[3])

        # crop img to original size
        crop_area = (upper_left, upper_left, lower_right, lower_right)
        return background.crop(crop_area)
    
    img = get_final_img(img)
        
    return img # returns cropped pil image