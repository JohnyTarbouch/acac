import numpy as np
from gym import Env, spaces

from .overcooked_MA_V1 import Overcooked_MA_V1


class Overcooked_MA_Meta_V1(Env):
    """
    Sample one Overcooked map type at the start of each episode.

    map_type can be "A,B,C", or a list/tuple such as. 
    For every reset, one of map types is selected
    uniformly at random, and all step/run calls for that 
    episode are forwarded to the selected real environment.

    The shared macro-action uses the A/B/C:

    - A has 14 macro-actions.
    - B and C have the same 14 macro-actions plus "go to counter".

    To keep the neural network output shape fixed, this meta environment always
    exposes 15 macro-actions. When the active map is A, "go to counter" is
    masked out in get_avail_agent_actions and translated to "stay".
    """

    metadata = Overcooked_MA_V1.metadata

    def __init__(
        self,
        grid_dim,
        task,
        rewardList,
        map_type="ABC",
        n_agent=2,
        obs_radius=2,
        mode="vector",
        debug=False,
        rand_start=False,
    ):
        # Keep one real environment per map type. Each env owns its own map,
        # agents, macro-action state, and item locations.
        self.map_types = self._parse_map_types(map_type)
        self.envs = {
            mt: Overcooked_MA_V1(
                grid_dim=grid_dim,
                task=task,
                rewardList=rewardList,
                map_type=mt,
                n_agent=n_agent,
                obs_radius=obs_radius,
                mode=mode,
                debug=debug,
                rand_start=rand_start,
            )
            for mt in self.map_types
        }

        # Use the first environment as the init. A real training
        # episode chooses the active in reset().
        self.active_map_type = self.map_types[0]
        self.active_env = self.envs[self.active_map_type]
        self.n_agent = self.active_env.n_agent

        # Shared action vocabulary exposed to the policy. Actions are translated
        # by name before being sent to the active real environment, because A and
        # B/C use different raw indices after "go to counter" is inserted.
        self.macroActionName = [
            "stay",
            "get tomato",
            "get lettuce",
            "get onion",
            "get plate 1",
            "get plate 2",
            "go to knife 1",
            "go to knife 2",
            "deliver",
            "chop",
            "go to counter",
            "right",
            "down",
            "left",
            "up",
        ]
        self.action_space = spaces.Discrete(len(self.macroActionName))
        self.observation_space = self.active_env.observation_space

        self._check_compatibility()

        # Translation tables:
        # - meta_to_env: policy action index -> active env action index
        # - env_to_meta: active env action index -> policy action index
        self._meta_to_env_action = {}
        self._env_to_meta_action = {}
        for mt, env in self.envs.items():
            meta_to_env = []
            env_to_meta = {}
            for env_idx, action_name in enumerate(env.macroActionName):
                env_to_meta[env_idx] = self.macroActionName.index(action_name)
            for action_name in self.macroActionName:
                if action_name in env.macroActionName:
                    meta_to_env.append(env.macroActionName.index(action_name))
                else:
                    meta_to_env.append(env.macroActionName.index("stay"))
            self._meta_to_env_action[mt] = meta_to_env
            self._env_to_meta_action[mt] = env_to_meta

    @staticmethod
    def _parse_map_types(map_type):
        #Normalize map_type config into a list of map labels
        if isinstance(map_type, (list, tuple)):
            map_types = list(map_type)
        elif "," in map_type:
            map_types = [mt.strip() for mt in map_type.split(",")]
        else:
            map_types = list(map_type)

        valid_map_types = {"A", "B", "C"}
        if not map_types or any(mt not in valid_map_types for mt in map_types):
            raise ValueError(f"map_type must contain only A, B, and C, got {map_type}")
        return map_types

    def _check_compatibility(self):
        """Fail early if the selected maps cannot share one policy model.

        A single actor/critic can only train across maps that agree on the
        model-facing dimensions. The raw action counts may differ only when the
        difference is covered by the shared macroActionName vocabulary.
        """
        reference_env = self.active_env
        reference_n_agent = reference_env.n_agent
        reference_obs_size = tuple(reference_env.obs_size)
        reference_state_size = reference_env.get_vector_state().shape[0]

        for mt, env in self.envs.items():
            if env.n_agent != reference_n_agent:
                raise ValueError(
                    f"Map {mt} has n_agent={env.n_agent}, expected {reference_n_agent}"
                )
            if tuple(env.obs_size) != reference_obs_size:
                raise ValueError(
                    f"Map {mt} has obs_size={env.obs_size}, expected {reference_obs_size}"
                )
            state_size = env.get_vector_state().shape[0]
            if state_size != reference_state_size:
                raise ValueError(
                    f"Map {mt} has state_size={state_size}, expected {reference_state_size}"
                )
            unknown_actions = [
                action_name
                for action_name in env.macroActionName
                if action_name not in self.macroActionName
            ]
            if unknown_actions:
                raise ValueError(
                    f"Map {mt} has unsupported macro-actions: {unknown_actions}"
                )

    @property
    def obs_size(self):
        return self.active_env.obs_size

    @property
    def state_size(self):
        # Centralized state dimension used by the critic
        return self.active_env.get_vector_state().shape[0]

    @property
    def n_action(self):
        # Shared action dimension for each agent
        return [self.action_space.n] * self.n_agent

    @property
    def action_spaces(self):
        return [self.action_space] * self.n_agent

    @property
    def macroAgent(self):
        # macro-agent state
        return self.active_env.macroAgent

    def reset(self):
        # Choose a map for the new episode and reset that real env
        self.active_map_type = np.random.choice(self.map_types)
        self.active_env = self.envs[self.active_map_type]
        return self.active_env.reset()

    def run(self, macro_actions):
        # Run one macro-action step 
        env_actions = self._to_env_actions(macro_actions)
        obs, rewards, terminate, info = self.active_env.run(env_actions)

        # Return action ids
        info["cur_mac"] = self._to_meta_actions(info["cur_mac"])
        info["map_type"] = self.active_map_type
        return obs, rewards, terminate, info

    def macro_action_sample(self):
        # Sample only currently available actions 
        avail_actions = self.get_avail_actions()
        return [
            int(np.random.choice(np.flatnonzero(avail_action)))
            for avail_action in avail_actions
        ]

    def get_vector_state(self):
        return self.active_env.get_vector_state()

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        env_avail_actions = self.active_env.get_avail_agent_actions(nth)
        meta_avail_actions = [0] * self.action_space.n
        env_to_meta = self._env_to_meta_action[self.active_map_type]
        for env_idx, available in enumerate(env_avail_actions):
            meta_avail_actions[env_to_meta[env_idx]] = available
        return meta_avail_actions

    def render(self, mode="human"):
        return self.active_env.render(mode=mode)

    def close(self):
        for env in self.envs.values():
            env.close()

    def _to_env_actions(self, macro_actions):
        # Translate shared meta action
        meta_to_env = self._meta_to_env_action[self.active_map_type]
        return [meta_to_env[action] if action >= 0 else action for action in macro_actions]

    def _to_meta_actions(self, macro_actions):
        # Translate active-env action ids
        env_to_meta = self._env_to_meta_action[self.active_map_type]
        return [env_to_meta[action] if action >= 0 else action for action in macro_actions]
