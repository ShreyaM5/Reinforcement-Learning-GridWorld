import random
import numpy as np

# ===========================
# Reward Constants
# ===========================
STEP_REWARD = -1
INVALID_MOVE_REWARD = -20
GOAL_REWARD = 100

# ===========================
# Action Constants
# ===========================
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


def generate_obstacles(
    n: int,
    num_obstacles: int,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> set[tuple[int, int]]:
    """
    Generate random obstacle positions.
    """

    if n <= 0:
        raise ValueError("Grid size must be positive.")

    if num_obstacles > (n * n - 2):
        raise ValueError(
            f"Maximum allowed obstacles are {n * n - 2}."
        )

    obstacles = set()

    while len(obstacles) < num_obstacles:

        row = random.randint(0, n - 1)
        col = random.randint(0, n - 1)

        if (row, col) == start:
            continue

        elif (row, col) == goal:
            continue

        elif (row, col) in obstacles:
            continue

        obstacles.add((row, col))

    return obstacles


class GridWorld:
    """
    Custom GridWorld environment.
    """

    def __init__(self, n: int, num_obstacles: int):

        self.n = n
        self.num_obstacles = num_obstacles

        self.grid = np.zeros((n, n), dtype=int)

        self.start = (0, 0)
        self.goal = (n - 1, n - 1)

        self.obstacles = generate_obstacles(
            self.n,
            self.num_obstacles,
            self.start,
            self.goal,
        )

        for row, col in self.obstacles:
            self.grid[row][col] = 1

        self.agent_position = self.start

    def render(self) -> None:
        """
        Print the current state of the GridWorld.
        """

        for row in range(self.n):
            for col in range(self.n):

                if (row, col) == self.agent_position and (row, col) == self.goal:
                    print("*", end=" ")

                elif (row, col) == self.agent_position:
                    print("A", end=" ")

                elif (row, col) == self.start:
                    print("S", end=" ")

                elif (row, col) == self.goal:
                    print("G", end=" ")

                elif self.grid[row][col] == 1:
                    print("X", end=" ")

                else:
                    print(".", end=" ")

            print()

    def reset(self) -> tuple[int, int]:
        """
        Reset the environment to the start state.
        """

        self.agent_position = self.start
        return self.agent_position

    def step(
        self, action: int
    ) -> tuple[tuple[int, int], int, bool]:
        """
        Execute one action.

        Returns:
            next_state, reward, done
        """

        row, col = self.agent_position

        if action == UP:
            new_row = row - 1
            new_col = col

        elif action == DOWN:
            new_row = row + 1
            new_col = col

        elif action == LEFT:
            new_row = row
            new_col = col - 1

        elif action == RIGHT:
            new_row = row
            new_col = col + 1

        else:
            raise ValueError("Invalid action.")

        # Invalid move (outside the grid)
        if not (0 <= new_row < self.n and 0 <= new_col < self.n):
            reward = INVALID_MOVE_REWARD
            done = False

        # Obstacle encountered
        elif (new_row, new_col) in self.obstacles:
            reward = INVALID_MOVE_REWARD
            done = False

        # Goal reached
        elif (new_row, new_col) == self.goal:
            reward = GOAL_REWARD
            done = True
            self.agent_position = (new_row, new_col)

        # Normal movement
        else:
            reward = STEP_REWARD
            done = False
            self.agent_position = (new_row, new_col)

        return self.agent_position, reward, done

    def state_to_index(self, position: tuple[int, int]) -> int:
        """
        Convert a grid position into a unique state index.
        """

        row, col = position
        return row * self.n + col