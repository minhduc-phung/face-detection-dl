import imutils


# Parameters:
# image: image we want to generate multiscale representations of
# scale: scale factor (how much the image is resized after each layer)
# minSize: minimum size of the output image
def image_pyramid(image, scale=1.5, min_size=(36, 36)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image
