from __future__ import print_function
import pygame
from pygame.locals import *
from misc.camera import Camera
from time import sleep, time
import numpy as np
import cv2


def runningMean(new, old, ratio):
    return ratio * new + (1. - ratio) * old


camera_device = 0
width = 640
height = 480

agility = .1

FPS = 30
border = 5

cam = Camera()
cam.setSize(width, height)
cam.capture(camera_device)

pygame.init()

fpsClock = pygame.time.Clock()

surface = pygame.display.set_mode((width, height), 0, 0)  # http://www.pygame.org/docs/ref/display.html#pygame.display.set_mode

pygame.display.set_caption('Control Console')
pygame.mouse.set_visible(1)

background = pygame.Surface(surface.get_size())
background.fill((20, 20, 20))

# test=pygame.image.load("test.png")

rect = pygame.Rect(0, 0, width, height)


border_time = 0.
forward = 0.
turn = 0.
keep_running = True
while keep_running:

    # surface.blit(background,(0,0))
    frame = cam.getFrame()
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    surface.blit(frame, (0, 0))

    reward = 0

    events = pygame.event.get()
    
    for event in events:
        if event.type == QUIT:
            keep_running = False
            print('Exiting!!!')

        if event.type == KEYDOWN:
            if (event.key == K_q):
                print('q')

            if event.key == K_EQUALS:
                border_time = time()
                border_color = [0, 255, 0]
                reward = 1

            elif event.key == K_MINUS:
                border_time = time()
                border_color = [255, 0, 0]
                reward = -1

            if event.key == K_ESCAPE:
                keep_running = False
                print('Exiting!!!')

    if time() - border_time < .2:
        pygame.draw.rect(surface, border_color, rect, border)

    keys_pressed = pygame.key.get_pressed()

    if keys_pressed[K_RIGHT]:
        turn = runningMean(1, turn, agility)

    elif keys_pressed[K_LEFT]:
        turn = runningMean(-1, turn, agility)

    else:
        turn = runningMean(0, turn, agility)

    if keys_pressed[K_UP]:
        forward = runningMean(1, forward, agility)

    elif keys_pressed[K_DOWN]:
        forward = runningMean(-1, forward, agility)

    else:
        forward = runningMean(0, forward, agility)

    if abs(turn) < .1:
        turn = 0.
    if abs(forward) < .1:
        forward = 0.

    if forward or turn or reward:
        print(reward, forward, turn)

    pygame.display.flip()  # or pygame.display.update()
    fpsClock.tick(FPS)

pygame.quit()
