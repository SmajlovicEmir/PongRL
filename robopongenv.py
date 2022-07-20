import gym
from gym import spaces
import numpy as np
from typing import Optional
from gym.envs.registration import register
import pygame


class RoboPongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, size: int = 5):
        import pygame
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.reward = int()
        'creates the paddle agent'
        self.paddle_agent = pygame.Surface((10, 100))
        self.paddle_agent.fill("Grey")
        self.paddle_agent_rect = self.paddle_agent.get_rect(topleft=(790, 100))
        'creates the target:ball'
        self.ball_target = pygame.Surface((10, 10))
        self.ball_target.fill("Red")
        self.ball_target_rect = self.ball_target.get_rect(topleft=(400, 200))
        'their positions on the screen'
        self._ball_pos = (self.ball_target_rect.x, self.ball_target_rect.y)
        self._paddle_pos = self.paddle_agent_rect.x
        self.window_sizex = 800
        self.window_sizey = 400
        'make a dummy paddle that always hits'
        self.dummy_paddle = pygame.Surface((10, 400))
        self.dummy_paddle.fill('Grey')
        self.dummy_paddle_rect = self.dummy_paddle.get_rect(topleft=(0, 0))
        'make ground rectangles'
        self.ground_surface1 = pygame.Surface((800, 10))
        self.ground_surface1.fill('Grey')
        self.ground_rect1 = self.ground_surface1.get_rect(topleft=(0, 390))
        self.ground_surface2 = pygame.Surface((800, 10))
        self.ground_surface2.fill('Grey')
        self.ground_rect2 = self.ground_surface2.get_rect(topleft=(0, 0))
        'determines ball velocity'
        self.ball_x_direction = 2
        self.ball_y_direction = 2

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.window_sizey, shape=(1,), dtype=int),
                "target": spaces.Box(0, self.window_sizex, shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(3)

        self._action_to_direction = {
            0: 1,
            1: 1,
            2: 0,
        }

        if render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_sizex, self.window_sizey))
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, return_info=False, options=None):
        self.window.fill(0)
        self.ball_target_rect.x = 400
        self.ball_target_rect.y = 200
        self.paddle_agent_rect.y = 100
        # self.reward = 0
        self._paddle_pos = np.array([self.paddle_agent_rect.y], dtype=int)
        self._ball_pos = np.array(([self.ball_target_rect.x, self.ball_target_rect.y]), dtype=int)
        self.ball_x_direction = 2
        self.ball_y_direction = 2
        observation = self._get_obs()
        info = self._get_info()
        return_info = False
        return (observation, info) if return_info else observation

    def _get_obs(self):
        return {"agent": self.paddle_agent_rect.y, "target": (self.ball_target_rect.y, self.ball_target_rect.x),
                "distancex": (self.ball_target_rect.x - self.paddle_agent_rect.x),
                "distancey": (self.ball_target_rect.y - self.paddle_agent_rect.y)}

    def _get_info(self):
        return {"top left paddle": self.paddle_agent_rect.topleft[1],
                "bot left paddle": self.paddle_agent_rect.bottomleft[1],
                "ball y": self.ball_target_rect.midright[1]}

    def step(self, action):
        self.window.fill(0)
        count = 0
        'keeps the paddle within the boundaries'
        if self.paddle_agent_rect.y >= 300:
            self.paddle_agent_rect.y -= 2
        if self.paddle_agent_rect.y <= 10:
            self.paddle_agent_rect.y += 2

        'moves the agent'
        if action == 0:
            self.paddle_agent_rect.y -= self._action_to_direction[0]
        elif action == 1:
            self.paddle_agent_rect.y += self._action_to_direction[1]
        elif action == 2:
            self.paddle_agent_rect.y += self._action_to_direction[2]

        'determines the direction of the ball'
        self.ball_target_rect.y += self.ball_y_direction
        self.ball_target_rect.x += self.ball_x_direction
        didhit = False

        if self.paddle_agent_rect.topleft[1] < self.ball_target_rect.midright[1] < self.paddle_agent_rect.bottomleft[1]:
            self.reward += 1
        else:
            self.reward -= 1

        'determines the collision'
        if self.paddle_agent_rect.colliderect(self.ball_target_rect):
            self.ball_x_direction = self.ball_x_direction * -1
            count += 1
            self.reward += 100

        if self.dummy_paddle_rect.colliderect(self.ball_target_rect):
            self.ball_x_direction = self.ball_x_direction * -1

        if self.ground_rect1.colliderect(self.ball_target_rect) or self.ground_rect2.colliderect(self.ball_target_rect):
            self.ball_y_direction = self.ball_y_direction * -1

        if count == 11:
            self.reward += 10000

        'did the opposite one score'
        done = False
        'negative reward for the loss of a point'
        if self.ball_target_rect.x <= 6 or self.ball_target_rect.x >= 790:
            self.reward -= 1000
            done = True

        observation = self._get_obs()
        info = self._get_info()
        self.render()
        reward = self.reward
        return observation, reward, done, info

    def render(self, mode="human"):
        import pygame
        self.window.blit(self.dummy_paddle, self.dummy_paddle_rect)
        self.window.blit(self.paddle_agent, self.paddle_agent_rect)
        self.window.blit(self.ball_target, self.ball_target_rect)
        self.window.blit(self.ground_surface1, self.ground_rect1)
        self.window.blit(self.ground_surface2, self.ground_rect2)
        pygame.display.update()
        self.clock.tick(120)

    def close(self):
        if self.window is not None:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            pygame.display.quit()
            pygame.quit()
