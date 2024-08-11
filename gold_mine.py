import numpy as np
import tkinter as tk

class GoldMineGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gold Mine Q-learning")

        self.rows_label = tk.Label(self.master, text="Number of Rows:")
        self.rows_label.grid(row=0, column=0)
        self.rows_entry = tk.Entry(self.master)
        self.rows_entry.grid(row=0, column=1)

        self.cols_label = tk.Label(self.master, text="Number of Columns:")
        self.cols_label.grid(row=1, column=0)
        self.cols_entry = tk.Entry(self.master)
        self.cols_entry.grid(row=1, column=1)

        self.submit_button = tk.Button(self.master, text="Submit", command=self.run_q_learning)
        self.submit_button.grid(row=2, column=0, columnspan=2)

        self.canvas = tk.Canvas(self.master)
        self.canvas.grid(row=3, column=0, columnspan=2)

        self.label = tk.Label(self.master, text="")
        self.label.grid(row=4, column=0, columnspan=2)

    def run_q_learning(self):
        # Retrieve the user-inputted matrix dimensions
        rows = int(self.rows_entry.get())
        cols = int(self.cols_entry.get())

        # Initialize the gold mine grid with random values
        gold_mine_grid = np.random.randint(1, 10, size=(rows, cols))

        # Parameters for Q-learning algorithm
        num_actions = 3  # 3 actions: move right, move diagonally up, move diagonally down
        Q = np.zeros((rows, cols, num_actions))  # Q-values for state-action pairs
        alpha = 0.1  # learning rate
        gamma = 0.9  # discount factor
        epsilon = 0.1  # exploration-exploitation trade-off

        # Q-learning algorithm
        num_episodes = 1000
        for episode in range(num_episodes):
            # Start a new episode, choose a random initial row
            row = np.random.randint(0, rows)
            col = 0

            while col < cols - 1:
                # Choose an action using epsilon-greedy policy
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.choice(range(num_actions))
                else:
                    action = np.argmax(Q[row, col])

                # Perform the chosen action and observe the reward and the next state
                if action == 0:
                    next_col = col + 1
                    next_row = row
                elif action == 1:
                    next_col = col + 1
                    next_row = row - 1 if row - 1 >= 0 else row
                else:
                    next_col = col + 1
                    next_row = row + 1 if row + 1 < rows else row

                reward = gold_mine_grid[next_row, next_col]
                Q[row, col, action] = (1 - alpha) * Q[row, col, action] + alpha * (
                        reward + gamma * np.max(Q[next_row, next_col, :]))

                # Move to the next state
                row, col = next_row, next_col

        # Calculate the maximum gold collected using the learned Q-values
        max_gold_collected = 0
        for row in range(rows):
            col = 0
            total_gold_collected = gold_mine_grid[row, col]
            while col < cols - 1:
                action = np.argmax(Q[row, col])
                if action == 0:
                    col += 1
                elif action == 1:
                    col += 1
                    row = max(row - 1, 0)
                else:
                    col += 1
                    row = min(row + 1, rows - 1)
                total_gold_collected += gold_mine_grid[row, col]
            max_gold_collected = max(max_gold_collected, total_gold_collected)

        # Display the gold mine grid in the Tkinter window
        self.display_gold_mine(gold_mine_grid)

        # Update the label with the maximum gold collected by the miner
        self.label.config(text=f"Max Gold Collected by the Miner: {max_gold_collected}")

    def display_gold_mine(self, gold_mine_grid):
        self.canvas.delete("all")  # Clear the canvas

        cell_size = 50
        for row in range(len(gold_mine_grid)):
            for col in range(len(gold_mine_grid[0])):
                x0, y0 = col * cell_size, row * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                color = self.get_color(gold_mine_grid[row, col])
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)
                self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=str(gold_mine_grid[row, col]))

    def get_color(self, value):
        # Map the gold value to a color (you can customize this mapping)
        color_map = {
            1: "#FFD700",  # Gold
            2: "#FFA500",  # Orange
            3: "#FF6347",  # Tomato
            4: "#32CD32",  # Lime Green
            5: "#4169E1",  # Royal Blue
            6: "#8A2BE2",  # Blue Violet
            7: "#FF1493",  # Deep Pink
            8: "#A52A2A",  # Brown
            9: "#228B22"   # Forest Green
        }
        return color_map.get(value, "#808080")  # Default to Gray for unknown values

def main():
    root = tk.Tk()
    gui = GoldMineGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()