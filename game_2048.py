import random

# Represents a model of the game 2048
class Game2048:


    # Initilize the game with a grid of given size and add two new tiles
    # grid_size: size of the grid with a default of 4
    def __init__(self,  grid_size = 4):
        self.grid_size = grid_size
        self.grid = [[0] * grid_size for _ in range(grid_size)]
        self.score = 0
        self.game_over = False
        self.add_new_tile()
        self.add_new_tile()


    # Include a new tile
    def add_new_tile(self):
        empty_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4


    # Moves all elements in row left
    def compress(self, row):
        new_row = [num for num in row if num != 0]
        new_row += [0] * (self.grid_size - len(new_row))
        return new_row


    # Merge all numbers that can be merged
    def merge(self, row):
        for i in range(self.grid_size - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row


    # Move row elements left
    def move_left(self):
        changed = False
        for i in range(self.grid_size):
            original = self.grid[i][:]
            self.grid[i] = self.compress(self.grid[i])
            self.grid[i] = self.merge(self.grid[i])
            self.grid[i] = self.compress(self.grid[i])
            if original != self.grid[i]:
                changed = True
        return changed


    # Flip grid horizontally 
    def reverse(self):
        self.grid = [row[::-1] for row in self.grid]


    # Swap rows and columns
    def transpose(self):
        self.grid = [list(row) for row in zip(*self.grid)]


    # Makes a move
    def move(self, direction):
        if self.game_over:
            return

        changed = False
        if direction == 'LEFT': # Move left using funciton
            changed = self.move_left()
            
        elif direction == 'RIGHT': # Move right by reversing a left move
            self.reverse()
            changed = self.move_left()
            self.reverse() 

        elif direction == 'UP': # Move up by transposing a left move 
            self.transpose()
            changed = self.move_left()
            self.transpose()

        elif direction == 'DOWN': # Move down by transposing and reversing a left move
            self.transpose()
            self.reverse()
            changed = self.move_left()
            self.reverse()
            self.transpose()

        if changed:
            self.add_new_tile()
            if not self.can_move():
                self.game_over = True


    # Check if any moves can be made
    def can_move(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 0:
                    return True
                if j < self.grid_size - 1 and self.grid[i][j] == self.grid[i][j + 1]:
                    return True
                if i < self.grid_size - 1 and self.grid[i][j] == self.grid[i + 1][j]:
                    return True
        return False


    # Reset the game to initial state
    def reset(self):
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.score = 0
        self.game_over = False
        self.add_new_tile()
        self.add_new_tile()
