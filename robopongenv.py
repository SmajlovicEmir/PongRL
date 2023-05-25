import gymnasium
from gymnasium import spaces
import numpy as np
from typing import Optional
import pygame
import random
from gymnasium.spaces import Discrete


class RoboPongEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, size: int = 5):
        import pygame
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.reward = int()
        pygame.font.init()
        self.count = 0
        'creates the paddle agent'
        self.paddle_agent = pygame.Surface((10, 100))
        self.paddle_agent.fill("Grey")
        self.paddle_agent_rect = self.paddle_agent.get_rect(topleft=(790, 200))
        'creates the target:ball'
        self.ball_target = pygame.Surface((10, 10))
        self.ball_target.fill("Red")
        self.ball_target_rect = self.ball_target.get_rect(topleft=(100, 100))
        'their positions on the screen'
        self._ball_pos = (self.ball_target_rect.x, self.ball_target_rect.y)
        self._paddle_pos = self.paddle_agent_rect.x
        self.window_sizex = 800
        self.window_sizey = 400
        'make a dummy paddle that always hits'
        self.dummy_paddle = pygame.Surface((10, self.window_sizey))
        self.dummy_paddle.fill('Grey')
        self.dummy_paddle_rect = self.dummy_paddle.get_rect(topleft=(0, 0))
        'make ground rectangle'
        self.ground_surface1 = pygame.Surface((self.window_sizex, 10))
        self.ground_surface1.fill('Grey')
        self.ground_rect1 = self.ground_surface1.get_rect(topleft=(0, 390))
        'ceiling'
        self.ground_surface2 = pygame.Surface((self.window_sizex, 10))
        self.ground_surface2.fill('Grey')
        self.ground_rect2 = self.ground_surface2.get_rect(topleft=(0, 0))
        'determines ball velocity'
        self.ball_x_direction = random.choice([-2, 2])
        self.ball_y_direction = random.choice([-2, 2])
        self.observation_space = spaces.Box(low = np.array([0, # lowest possible observation paddle
                                                      0, # lowest possible observation ball.x
                                                      0, # lowest possible observation ball.x
                                                      -self.window_sizex, # lowest possible observation delta x
                                                      -self.window_sizey # lowest possible observation delta.y
                                                      ], dtype=np.float32),
                                            high =np.array([self.window_sizey, # highest possible observation paddle.y
                                                      self.window_sizex, # highest possible observation ball.x
                                                      self.window_sizey, # highest possible observation ball.y
                                                      0, # highest possible observation delta.x
                                                      self.window_sizey] # highest possible observation delta.y
                                                     , dtype=np.float32),
                                            dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self._action_to_direction = {
            0: -1 if 10 <= self.paddle_agent_rect.topleft[1] else 0,  # zero for up
            1: 1 if self.paddle_agent_rect.bottomleft[1] <= self.window_sizey - self.dummy_paddle_rect.top else 0,
            # one for down
        }
        if render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_sizex, self.window_sizey))
        self.clock = pygame.time.Clock()
        self.info = {"paddle y": 0,
                     "ball x ": 0,
                     "ball y ": 0,
                     "delta x ": 0,
                     "delta y ": 0,
                     }
        self.key_list = list(self.info)

    def _get_obs(self):
        self.delta_x = (self.ball_target_rect.midright[0] - self.paddle_agent_rect.midright[0])
        self.delta_y = self.ball_target_rect.midright[1] - self.paddle_agent_rect.midleft[1]
        self.obs_array = [self.paddle_agent_rect.y, # vertical alignment of the paddle
                          self.ball_target_rect.x, # horizontal position of the ball
                          self.ball_target_rect.y, # vertical position of the ball
                          self.delta_x, # difference between paddle x and ball x
                          self.delta_y] # difference between paddle y and ball y
        return np.array(self.obs_array, dtype=np.float32)

    def _get_info(self):
        for key,obs in zip(self.key_list, self._get_obs()):
            self.info[key] = obs
        return self.info

    def step(self, action):
        self.window.fill(0)

        # this is to prevent crashing when clicking away from the screen
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                None

        # setup of the action space
        self._action_to_direction = {
            0: -1 if self.ground_rect2.bottom <= self.paddle_agent_rect.topleft[1] else 1, # zero for up
            1: 1 if self.paddle_agent_rect.bottomleft[1] + 10 <= self.window_sizey - self.ground_rect2.top else 0, # one for down
        }
        self.reward = 0

        # moves the agent (paddle)
        self.paddle_agent_rect.y += self._action_to_direction[action]

        # determines the direction of the ball
        self.ball_target_rect.y += self.ball_y_direction
        self.ball_target_rect.x += self.ball_x_direction

        # gives a reward if the ball is within the upper and lower boundaries of paddle.y
        # this is to decrease the sparsness of the rewards
        if self.paddle_agent_rect.topleft[1] < self.ball_target_rect.midright[1] < self.paddle_agent_rect.bottomleft[1]:
            self.reward = 1
        else:
            self.reward = -1

        # determines the collision

        if self.paddle_agent_rect.colliderect(self.ball_target_rect):
            self.ball_target_rect.x -= 4
            self.ball_x_direction = self.ball_x_direction * -1
            self.count += 1
            self.reward = 100

        # changes the balls direction if it hits one of the boundaries or the paddle
        if self.dummy_paddle_rect.colliderect(self.ball_target_rect):
            self.ball_x_direction = self.ball_x_direction * -1

        if self.ground_rect1.colliderect(self.ball_target_rect) or self.ground_rect2.colliderect(self.ball_target_rect):
            self.ball_y_direction = self.ball_y_direction * -1

        #additional reward for scoring a multiple of 21 points
        if self.count % 21 == 0 and self.count > 0 :
            self.reward = 10000


        # gives a negative reward for losing a point, alse determines if the episode is over
        terminated = False
        if self.ball_target_rect.x >= 790:
            self.reward = -100
            terminated = True

        # for return purposes, returns the observations in both cases, info can be used for debugging purposes
        observation = self._get_obs()
        info = self._get_info()

        return observation, self.reward, terminated, False, info

    def reset(self, seed=None, return_info=False, options=None):
        self.window.fill(0)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                None
        self.count = 0
        self.ball_target_rect.x = self.window_sizex/2
        self.ball_target_rect.y = self.window_sizey/2
        self.paddle_agent_rect.y = self.window_sizey/2
        self.reward = 0
        self.ball_x_direction = random.choice([-2, 2])
        self.ball_y_direction = random.choice([-2, 2])
        observation = self._get_obs()
        info = self._get_info()
        return_info = False
        return (observation, info) if return_info else observation

    def render(self):
        import pygame
        self.window.blit(self.dummy_paddle, self.dummy_paddle_rect)
        self.window.blit(self.paddle_agent, self.paddle_agent_rect)
        self.window.blit(self.ball_target, self.ball_target_rect)
        self.window.blit(self.ground_surface1, self.ground_rect1)
        self.window.blit(self.ground_surface2, self.ground_rect2)
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        if self.window is not None:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            pygame.display.quit()
            pygame.quit()
