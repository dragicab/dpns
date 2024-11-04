import cv2

image_path = 'fingerprint1.jpg'
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def generate_descriptor(image):
    orb = cv2.ORB_create()
    keypoints, descriptor = orb.detectAndCompute(image, None)
    return keypoints, descriptor


def compare_descriptors(descriptor1, descriptor2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)
    similarity_score = sum(match.distance for match in matches) / len(matches)
    return similarity_score.__trunc__()


def main():
    image1 = preprocess_image('fingerprint1.jpg')
    image2 = preprocess_image('fingerprint2.jpg')

    keypoints1, descriptor1 = generate_descriptor(image1)
    keypoints2, descriptor2 = generate_descriptor(image2)

    similarity_score = compare_descriptors(descriptor1, descriptor2)
    similarity_string = "{}%".format(similarity_score)
    print("Descriptor for image 1:")
    print(descriptor1)

    print("\nDescriptor for image 2:")
    print(descriptor2)

    print("\nSimilarity:", similarity_string)
if __name__ == "__main__":
    main()

