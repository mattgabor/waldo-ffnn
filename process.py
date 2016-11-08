from PIL import Image
im = Image.open('data/64/waldo/1_4_6.jpg')
pix_val = list(im.getdata())
