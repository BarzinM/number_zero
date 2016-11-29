import pygame
from pygame.locals import *

pygame.init()

FPS = 30
fpsClock = pygame.time.Clock()

width = 0
height = 0
surface = pygame.display.set_mode((width, height), 0, 0) # http://www.pygame.org/docs/ref/display.html#pygame.display.set_mode
pygame.display.set_caption('Control Console')
pygame.mouse.set_visible(0)

keep_running = True
#test=pygame.image.load("test.png")
test=pygame.image.load("test.png")
test2=pygame.image.load("test2.png")

while keep_running:
    events = pygame.event.get()
    # print(events)
    for event in events:
        if event.type == QUIT:
            keep_running = False
            print('Exiting!!!')
            # sys.exit()

        if event.type == KEYDOWN:
            if (event.key == K_q):
                print('q')
                surface.blit(test,(0,0))
    #         elif (event.key == K_RIGHT):
    #             print('right')
    #         elif (event.key == K_UP):
    #             print('up')
    #         elif (event.key == K_DOWN):
    #             print('down')

    keys_pressed = pygame.key.get_pressed()

    forward = 0
    turn = 0

    if keys_pressed[K_LEFT]:
        turn = -1

    if keys_pressed[K_RIGHT]:
        turn = 1

    if keys_pressed[K_UP]:
        forward = 1

    if keys_pressed[K_DOWN]:
        forward = -1

    if keys_pressed[K_e]:
        print('e')
        surface.blit(test2,(0,0))

    if keys_pressed[K_ESCAPE]:
        keep_running = False
        print('Exiting!!!')

    if forward or turn:
        print(forward, turn)

    pygame.display.update()
    fpsClock.tick(FPS)

pygame.quit()
