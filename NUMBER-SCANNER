import pygame
import pygame.camera
from pygame.locals import *
from PIL import Image
import pytesseract
import numpy as np

pygame.init()
pygame.camera.init()

def capture_image():
    camlist = pygame.camera.list_cameras()
    if not camlist:
        raise Exception("No camera found")

    cam = pygame.camera.Camera(camlist[0], (640, 480))
    cam.start()

    print("Press 'c' to capture an image or 'q' to quit.")
    while True:
        image = cam.get_image()
        pygame.display.set_mode((640, 480))
        screen = pygame.display.get_surface()
        screen.blit(image, (0, 0))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    cam.stop()
                    pygame.quit()
                    return pygame.surfarray.array3d(image)
                elif event.key == pygame.K_q:
                    cam.stop()
                    pygame.quit()
                    return None

def preprocess_image(img):
    img = Image.fromarray(img)
    img = img.convert('L')
    img = img.resize((640, 480), Image.ANTIALIAS)
    return img

def recognize_text(img):
    text = pytesseract.image_to_string(img)
    return text

def main():
    captured_img = capture_image()
    if captured_img is None:
        print("No image captured. Exiting.")
        return

    processed_img = preprocess_image(captured_img)
    recognized_text = recognize_text(processed_img)
    print(f"Recognized Text: {recognized_text}")

if __name__ == "__main__":
    main()
