import pygame
import random

# vars ??
WIDTH = 800
HEIGHT = 400

## starting pos not in center of screen, but can be changed to center if needed
fly_x = 100
fly_y = 200

## flyspeed may change higher/lower depending on the tests
fly_speed = 3
fly_radius = 10

## action variables (how it moves left/right/nothing)
action = 1
action_timer = 0
action_hold = 15

## goal (on the right side of the screen, can be changed to anywhere)
goal_x = 700
goal_y = 200
goal_radius = 15

## reward tracking
reward = 0
episode_reward = 0


pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fly Brain Testing Environment")

clock = pygame.time.Clock()


# old keyboard testing code for later
# keys = pygame.key.get_pressed()
#
# if keys[pygame.K_LEFT]:
#     fly_x = fly_x - fly_speed
#
# if keys[pygame.K_RIGHT]:
#     fly_x = fly_x + fly_speed


running = True

while running:

    old_distance = abs(goal_x - fly_x)

    # for window closing
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # counts how long we have been doing current action
    action_timer = action_timer + 1

    # picks a new action every 15 frames
    if action_timer >= action_hold:
        action = random.randint(0, 2)
        action_timer = 0

    # 0 = left
    # 1 = do nothing
    # 2 = right
    if action == 0:
        fly_x = fly_x - fly_speed

    if action == 1:
        pass

    if action == 2:
        fly_x = fly_x + fly_speed

    # window boundaries
    if fly_x < fly_radius:
        fly_x = fly_radius

    if fly_x > WIDTH - fly_radius:
        fly_x = WIDTH - fly_radius

    new_distance = abs(goal_x - fly_x)

    # reward if fly moves closer to goal
    if new_distance < old_distance:
        reward = 0.1

    elif new_distance > old_distance:
        reward = -0.1

    else:
        reward = 0

    episode_reward = episode_reward + reward

    # check if fly reached the goal
    if new_distance <= fly_radius + goal_radius:

        reward = 10
        episode_reward = episode_reward + reward

        print("GOAL REACHED")
        print("Episode reward:", round(episode_reward, 2))

        # reset fly
        fly_x = 100

        # reset action stuff
        action = 1
        action_timer = 0

        # reset reward for next attempt
        episode_reward = 0

    # background color
    screen.fill((30, 30, 30))

    # goal is green circle
    pygame.draw.circle(
        screen,
        (0, 255, 0),
        (goal_x, goal_y),
        goal_radius
    )

    # fly is yellow circle lol
    pygame.draw.circle(
        screen,
        (255, 255, 0),
        (fly_x, fly_y),
        fly_radius
    )

    pygame.display.flip()

    clock.tick(60)


pygame.quit()