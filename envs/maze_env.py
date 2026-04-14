import re
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from mazelib import Maze
    from mazelib.generate.Prims import Prims
    MAZELIB_AVAILABLE = True
except Exception:
    MAZELIB_AVAILABLE = False

MAZE_LAYOUTS = {
    "Maze-Easy": [
        "########",
        "#S.....#",
        "#..##..#",
        "#..##..#",
        "#......#",
        "#.####.#",
        "#.....G#",
        "########",
    ],
    "Maze-Medium": [
        "########",
        "#S.#...#",
        "#.#.##.#",
        "#...#..#",
        "###.#..#",
        "#...##.#",
        "#..#..G#",
        "########",
    ],
    "Maze-Hard": [
        "########",
        "#S.#.#.#",
        "#.#.#.#.#",
        "#...#...#",
        "###.###.#",
        "#......#",
        "#.####G#",
        "########",
    ],
    "Maze-Stage": [
        "########",
        "#S...#.#",
        "#.##.#.#",
        "#..#...#",
        "#..###.#",
        "#......#",
        "#..##.G#",
        "########",
    ],
}


def parse_auto_maze_id(layout_id: str) -> tuple[int, int] | None:
    if layout_id == "Maze-Auto":
        return 9, 9
    match = re.match(r"Maze-Auto-(\d+)x(\d+)$", layout_id)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    return None


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value - 1 if value > 3 else 3


def generate_auto_maze_layout(width: int, height: int, seed: int | None = None) -> list[str]:
    width = _ensure_odd(width)
    height = _ensure_odd(height)
    if MAZELIB_AVAILABLE:
        try:
            maze = Maze()
            maze.generator = Prims(width, height)
            maze.generate()
            grid = np.array(maze.grid, dtype=int)
            layout = ["".join("#" if cell else "." for cell in row) for row in grid]
            if len(layout) != height or any(len(row) != width for row in layout):
                raise ValueError("mazelib returned wrong maze dimensions")
        except Exception:
            layout = _generate_perfect_maze(width, height, seed)
    else:
        layout = _generate_perfect_maze(width, height, seed)
    layout = [list(row) for row in layout]
    layout[1][1] = "S"
    layout[height - 2][width - 2] = "G"
    return ["".join(row) for row in layout]


def _generate_perfect_maze(width: int, height: int, seed: int | None = None) -> list[str]:
    rng = np.random.default_rng(seed)
    grid = np.full((height, width), "#", dtype='<U1')
    visited = np.zeros(((height - 1) // 2, (width - 1) // 2), dtype=bool)

    def cell_to_pos(cy: int, cx: int) -> tuple[int, int]:
        return 1 + cy * 2, 1 + cx * 2

    def carve(cy: int, cx: int):
        visited[cy, cx] = True
        y, x = cell_to_pos(cy, cx)
        grid[y, x] = "."

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        rng.shuffle(directions)

        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < visited.shape[0] and 0 <= nx < visited.shape[1] and not visited[ny, nx]:
                wy, wx = y + dy, x + dx
                grid[wy, wx] = "."
                carve(ny, nx)

    carve(0, 0)
    return ["".join(row) for row in grid]


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        layout_id: str = "Maze-Easy",
        max_steps: int = 200,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        self.layout_id = layout_id
        self.auto_size = parse_auto_maze_id(layout_id)
        self.auto_generate = self.auto_size is not None

        if self.auto_generate:
            self.width, self.height = self.auto_size
            self.width = _ensure_odd(self.width)
            self.height = _ensure_odd(self.height)
            self._base_layout = None
        else:
            if layout_id not in MAZE_LAYOUTS:
                raise ValueError(f"Unknown maze layout: {layout_id}")
            self._base_layout = [list(row) for row in MAZE_LAYOUTS[layout_id]]
            self.height = len(self._base_layout)
            self.width = len(self._base_layout[0])

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.np_random = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        self.step_count = 0
        self.agent_pos = (1, 1)
        self.goal_pos = (self.height - 2, self.width - 2)
        self.grid = None

        if self.auto_generate:
            layout = generate_auto_maze_layout(self.width, self.height, seed=seed)
            self._base_layout = [list(row) for row in layout]

        self.reset(seed=seed)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        if self.auto_generate:
            self.grid = np.array(self._base_layout, dtype='<U1')
        elif self.layout_id == "Maze-Stage":
            selected = self.np_random.choice(["Maze-Easy", "Maze-Medium", "Maze-Hard"])
            self.grid = np.array([list(row) for row in MAZE_LAYOUTS[selected]], dtype='<U1')
        else:
            self.grid = np.array(self._base_layout, dtype='<U1')

        self.agent_pos = self._find_symbol("S")
        self.goal_pos = self._find_symbol("G")
        self._clear_symbols()

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        self.step_count += 1
        next_pos = self._next_position(action)
        if self._is_free(next_pos):
            self.agent_pos = next_pos

        terminated = self.agent_pos == self.goal_pos
        truncated = self.step_count >= self.max_steps
        reward = 1.0 if terminated else 0.0

        obs = self._get_observation()
        info = {}
        return obs, reward, terminated, truncated, info

    def _find_symbol(self, symbol: str) -> tuple[int, int]:
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == symbol:
                    return (y, x)
        raise ValueError(f"Symbol {symbol} not found in maze layout")

    def _clear_symbols(self):
        self.grid[self.grid == "S"] = "."
        self.grid[self.grid == "G"] = "."

    def _next_position(self, action: int) -> tuple[int, int]:
        y, x = self.agent_pos
        if action == 0:
            return y - 1, x
        if action == 1:
            return y, x + 1
        if action == 2:
            return y + 1, x
        if action == 3:
            return y, x - 1
        return y, x

    def _is_free(self, pos: tuple[int, int]) -> bool:
        y, x = pos
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return False
        return self.grid[y, x] != "#"

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == "#":
                    obs[y, x, 0] = 1
        goal_y, goal_x = self.goal_pos
        obs[goal_y, goal_x, 1] = 1
        agent_y, agent_x = self.agent_pos
        obs[agent_y, agent_x, 2] = 1
        return obs

    def _get_render_frame(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        frame[self.grid == "#"] = [80, 80, 80]
        goal_y, goal_x = self.goal_pos
        frame[goal_y, goal_x] = [0, 200, 0]
        agent_y, agent_x = self.agent_pos
        frame[agent_y, agent_x] = [0, 0, 255]
        return frame

    def render(self, mode: str = "human"):
        frame = self._get_render_frame()
        if mode == "rgb_array":
            return frame
        if mode == "human":
            try:
                import matplotlib.pyplot as plt
                plt.imshow(frame)
                plt.axis("off")
                plt.pause(0.001)
            except Exception:
                display = []
                for y in range(self.height):
                    row = []
                    for x in range(self.width):
                        if (y, x) == self.agent_pos:
                            row.append("A")
                        elif (y, x) == self.goal_pos:
                            row.append("G")
                        elif self.grid[y, x] == "#":
                            row.append("#")
                        else:
                            row.append(".")
                    display.append("".join(row))
                print("\n".join(display))
                print(f"step={self.step_count} pos={self.agent_pos} reward_goal={self.agent_pos == self.goal_pos}")

    def close(self):
        return None


def make_maze_env(env_id: str = "Maze-Easy", seed: int | None = None) -> MazeEnv:
    return MazeEnv(layout_id=env_id, seed=seed)


__all__ = ["MazeEnv", "make_maze_env"]
