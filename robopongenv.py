import gymnasium
from gymnasium.spaces import Discrete
import numpy as np
from typing import Optional
import random


class RoboPongEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, size: int = 5):
        import pygame
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.reward = int()
        self.rewhist = []
        self.rewhistnp = np.array(self.rewhist)
        pygame.font.init()
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
        self.dummy_paddle = pygame.Surface((10, 400))
        self.dummy_paddle.fill('Grey')
        self.dummy_paddle_rect = self.dummy_paddle.get_rect(topleft=(0, 0))
        'make ground rectangles'
        self.ground_surface1 = pygame.Surface((800, 10))
        self.ground_surface1.fill('Grey')
        self.ground_rect1 = self.ground_surface1.get_rect(topleft=(0, 390))
        'ceiling'
        self.ground_surface2 = pygame.Surface((800, 10))
        self.ground_surface2.fill('Grey')
        self.ground_rect2 = self.ground_surface2.get_rect(topleft=(0, 0))
        'determines ball velocity'
        self.ball_x_direction = random.choice([-2, 2])
        self.ball_y_direction = random.choice([-2, 2])
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, -1 * self.window_sizey], dtype=np.float32),
                                            np.array([self.window_sizey, self.window_sizex, self.window_sizey,
                                                      self.window_sizex, self.window_sizey], dtype=np.float32),
                                            dtype=np.float32)
        """
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.window_sizey, shape=(1,), dtype=int),
                "target": spaces.Box(0, self.window_sizex, shape=(2,), dtype=int),
            }
        )
        """
        dummy = Discrete(2)
        print(type(dummy))

        self.action_space = gymnasium.spaces.Discrete(2)
        self._action_to_direction = {
            0: 1 if 100 <= self.paddle_agent_rect.y else 0,
            1: 1 if 100 <= self.paddle_agent_rect.y <= 290 else 0,
        }
        if render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_sizex, self.window_sizey))
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, return_info=False, options=None):
        self.window.fill(0)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                None
        count = 0
        self.ball_target_rect.x = 400
        self.ball_target_rect.y = 200
        self.paddle_agent_rect.y = 100
        self.temp_reward = 0
        # self.reward = 0
        self._paddle_pos = np.array([self.paddle_agent_rect.y], dtype=np.float32)
        self._ball_pos = np.array(([self.ball_target_rect.x, self.ball_target_rect.y]), dtype=np.float32)
        self.ball_x_direction = random.choice([-2, 2])
        self.ball_y_direction = random.choice([-2, 2])
        observation = self._get_obs()
        info = self._get_info()
        return_info = False
        return (observation, info) if return_info else observation

    def _get_obs(self):
        self.delta_x = abs(self.ball_target_rect.midright[0] - self.paddle_agent_rect.midright[0])
        self.delta_y = self.ball_target_rect.midright[1] - self.paddle_agent_rect.midleft[1]
        self.obs_array = [self.paddle_agent_rect.x, self.ball_target_rect.x, self.ball_target_rect.y, self.delta_x,
                          self.delta_y]
        return np.array(self.obs_array, dtype=np.float32)

    def _get_info(self):
        return {"paddle y": self.paddle_agent_rect.y,
                "ball y": self.ball_target_rect.y,
                "ball x": self.ball_target_rect.x}

    def step(self, action):
        self.window.fill(0)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                None
        self._action_to_direction = {
            0: 1 if 10 <= self.paddle_agent_rect.y else 0,
            1: 1 if self.paddle_agent_rect.y <= 300 else 0,
        }

        count = 0
        'keeps the paddle within the boundaries'
        """
        if self.paddle_agent_rect.y >= 300:
            self.paddle_agent_rect.y -= 2
        if self.paddle_agent_rect.y <= 10:
            self.paddle_agent_rect.y += 2
        """
        'moves the agent'
        if action == 0:
            self.paddle_agent_rect.y -= self._action_to_direction[0]
        elif action == 1:
            self.paddle_agent_rect.y += self._action_to_direction[1]
        # elif action == 2:
        #    self.paddle_agent_rect.y += self._action_to_direction[2]

        'determines the direction of the ball'
        self.ball_target_rect.y += self.ball_y_direction
        self.ball_target_rect.x += self.ball_x_direction

        'reward if ball within paddle boundary'
        """
        if self.paddle_agent_rect.topleft[1] < self.ball_target_rect.midright[1] < self.paddle_agent_rect.bottomleft[1]:
            self.reward += 1
            self.rewhist.append(self.reward)
        else:
            self.reward -= 1
            self.rewhist.append(self.reward)
        """
        'determines the collision'

        observation = self._get_obs()
        info = self._get_info()

        if self.paddle_agent_rect.colliderect(self.ball_target_rect):
            self.ball_target_rect.x -= 4
            self.ball_x_direction = self.ball_x_direction * -1
            count += 1
            self.temp_reward += 100
            self.rewhist.append(self.temp_reward)
            if len(self.rewhist) > 2:
                self.rewhistnp = np.array(self.rewhist)
                self.reward = (self.temp_reward - self.rewhistnp.mean()) / np.std(self.rewhistnp)

        if self.dummy_paddle_rect.colliderect(self.ball_target_rect):
            self.ball_x_direction = self.ball_x_direction * -1

        if self.ground_rect1.colliderect(self.ball_target_rect) or self.ground_rect2.colliderect(self.ball_target_rect):
            self.ball_y_direction = self.ball_y_direction * -1
        """
        if count == 5:
            self.reward += 10000
            self.rewhist.append(self.reward)
        """
        'did the opposite one score'
        terminated = False

        'negative reward for the loss of a point'
        if self.ball_target_rect.x >= 790:
            self.temp_reward -= 100
            self.rewhist.append(self.temp_reward)
            if len(self.rewhist) > 2:
                self.rewhistnp = np.array(self.rewhist)
                self.reward = (self.temp_reward - self.rewhistnp.mean()) / (np.std(self.rewhistnp) + 1e-10)
            terminated = True
            """
            if done:
                self.reset()
            """

        # print(f"rewhistnpstd: {np.std(self.rewhistnp)}")

        # self.render()
        truncated = False
        return observation, self.reward, terminated, truncated, info

    def render(self, mode="human"):
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
