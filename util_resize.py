import cv2

# resize the image to a fixed size
def resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    img = cv2.imread("imgs/Eiffel_Tower_resized.jpg")
    img_resized = resize(img, (400, 316))
    cv2.imshow("Resized Image", img_resized)
    cv2.imwrite("imgs/Eiffel_Tower_resized.jpg", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()