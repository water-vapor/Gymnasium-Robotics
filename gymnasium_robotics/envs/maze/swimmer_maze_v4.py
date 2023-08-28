"""A maze environment with the Gymnasium Swimmer agent (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/swimmer_v4.py).

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve reusing the code in Gymnasium for the Swimmer environment and in `point_maze/maze_env.py`.
The new code also follows the Gymnasium API and Multi-goal API

This project is covered by the Apache 2.0 License.
"""

import sys
from os import path
from typing import Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.swimmer_v5 import SwimmerEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.maze.maps import U_MAZE
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


class SwimmerMazeEnv(MazeEnv, EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        **kwargs,
    ):
        # Get the swimmer.xml path from the Gymnasium package
        swimmer_xml_file_path = path.join(
            path.dirname(sys.modules[SwimmerEnv.__module__].__file__), "assets/swimmer.xml"
        )
        super().__init__(
            agent_xml_path=swimmer_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=4,
            maze_height=0.5,
            reward_type=reward_type,
            continuing_task=continuing_task,
            **kwargs,
        )
        # Create the MuJoCo environment, include position observation of the Swimmer for GoalEnv
        self.swimmer_env = SwimmerEnv(
            xml_file=self.tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            reset_noise_scale=0.0,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.swimmer_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.swimmer_env.action_space
        obs_shape: tuple = self.swimmer_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape[0] - 2,), dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            render_mode,
            maze_map,
            reward_type,
            continuing_task,
            **kwargs,
        )

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.swimmer_env.init_qpos[:2] = self.reset_pos

        obs, info = self.swimmer_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        return obs_dict, info

    def step(self, action):
        swimmer_obs, _, _, _, info = self.swimmer_env.step(action)
        obs = self._get_obs(swimmer_obs)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        info["success"] = bool(np.linalg.norm(obs["achieved_goal"] - self.goal) <= 0.45)

        if self.render_mode == "human":
            self.render()

        # Update the goal position if necessary
        self.update_goal(obs["achieved_goal"])

        return obs, reward, terminated, truncated, info

    def _get_obs(self, swimmer_obs: np.ndarray) -> Dict[str, np.ndarray]:
        achieved_goal = swimmer_obs[:2]
        observation = swimmer_obs[2:]

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def update_target_site_pos(self):
        self.swimmer_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def render(self):
        return self.swimmer_env.render()

    def close(self):
        super().close()
        self.swimmer_env.close()
