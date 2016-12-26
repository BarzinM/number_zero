from __future__ import print_function
import pygame
from pygame.locals import *
from lib.camera import Camera, writeFrame
from time import sleep, time
import numpy as np
import cv2
import glob
import os
from sys import stdout
import struct
import socket


def runningMean(new, old, ratio):
    return ratio * new + (1. - ratio) * old


address = ("192.168.1.4", 8089)
width = 640
height = 480

agility = .1

FPS = 30
border = 15


def startConsole(directory=None, camera_address=None):

    if directory is None:
        should_save = False
    else:
        should_save = True

        if not os.path.isdir(directory):
            os.makedirs(directory)

        dir_positive = os.path.join(directory, 'positives')
        if not os.path.isdir(dir_positive):
            os.makedirs(dir_positive)
        files_positive = glob.glob(os.path.join(dir_positive, '*.png'))
        if files_positive:
            files_positive = [os.path.split(file)[1]
                              for file in files_positive]
            files_positive = [int(os.path.splitext(file)[0])
                              for file in files_positive]
            id_positive = sorted(files_positive)[-1] + 1
        else:
            id_positive = 0

        dir_negative = os.path.join(directory, 'negatives')
        if not os.path.isdir(dir_negative):
            os.makedirs(dir_negative)
        files_negative = glob.glob(os.path.join(dir_negative, '*.png'))
        if files_negative:
            files_negative = [os.path.split(file)[1]
                              for file in files_negative]
            files_negative = [int(os.path.splitext(file)[0])
                              for file in files_negative]
            id_negative = sorted(files_negative)[-1] + 1
        else:
            id_negative = 0

        print("Starting positive id from %i and negative id from %i." %
              (id_positive, id_negative))

    cam = Camera()

    if camera_address is None:
        send_command = False
        cam.capture()
    else:
        send_command = True
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        connection.settimeout(100)
        connection.connect(camera_address)
        cam.receive2(connection) # TODO: refactore camera module

    pygame.init()

    fpsClock = pygame.time.Clock()

    # http://www.pygame.org/docs/ref/display.html#pygame.display.set_mode
    surface = pygame.display.set_mode((width, height), 0, 0)

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
        frame_received = cam.getFrame()
        frame = cv2.resize(frame_received, (width, height))
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
                    # save to positive directory
                    if should_save:
                        # writeFrame(frame_received, os.path.join(
                        #     dir_positive, '%i' % id_positive))
                        id_positive += 1
                elif event.key == K_MINUS:
                    border_time = time()
                    border_color = [255, 0, 0]
                    reward = -1
                    # save to negative directory
                    if should_save:
                        # writeFrame(frame_received, os.path.join(
                        #     dir_negative, '%i' % id_negative))
                        id_negative += 1

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
            _turn = 0.
        else:
            _turn = turn
        if abs(forward) < .1:
            _forward = 0.
        else:
            _forward = forward

        message = struct.pack(">cffc", "s",_forward,_turn,"e")
        # print(struct.unpack(">cffc",message))

        if send_command:
            connection.send(message)

        # if forward or turn or reward:
        print("%+i | %+ 1.3f | %+ 1.3f"%(reward, forward, turn))
        stdout.write("\033[F\033[K")

        pygame.display.flip()  # or pygame.display.update()
        fpsClock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    startConsole(directory='/tmp/console',camera_address=address)
