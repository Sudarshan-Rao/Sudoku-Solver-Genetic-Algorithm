import time
from copy import deepcopy
from random import sample
import numpy as np
import sudoku_BT as btss
import sudoku_GA as gss


def generate_sudoku(n):
    side = n*n
    nums = sample(range(1, side + 1), side)  # random numbers
    board = [[nums[(n * (r % n) + r // n + c) % side] for c in range(side)] for r in range(side)]
    rows = [r for g in sample(range(n), n) for r in sample(range(g * n, (g + 1) * n), n)]
    cols = [c for g in sample(range(n), n) for c in sample(range(g * n, (g + 1) * n), n)]
    board = [[board[r][c] for c in cols] for r in rows]
    return board

def generate_removal_map(solution_board):
    board = deepcopy(solution_board)
    side_length = len(board)
    removal_map = [[0 for _ in range(side_length)] for _ in range(side_length)]
    squares = side_length * side_length
    empty = squares * 3//4
    for p in sample(range(squares), empty):
        board[p // side_length][p % side_length] = 0
        removal_map[p // side_length][p % side_length] = 1
    return board, removal_map

def pretty_print(board):
    side = len(board)
    base = int(side ** 0.5)
    expand_line = lambda line : line[0]+line[5:9].join([line[1:5]*(base-1)]*base)+line[9:13]
    line0 = expand_line("╔═══╤═══╦═══╗")
    line1 = expand_line("║ . │ . ║ . ║")
    line2 = expand_line("╟───┼───╫───╢")
    line3 = expand_line("╠═══╪═══╬═══╣")
    line4 = expand_line("╚═══╧═══╩═══╝")
    symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums = [[""] + [symbol[n] for n in row] for row in board]
    print(line0)
    for r in range(1, side + 1):
        print("".join(n + s for n, s in zip(nums[r - 1], line1.split("."))))
        print([line2, line3, line4][(r % side == 0) + (r % base == 0)])

#################################################################################################################

if __name__ == "__main__":
    # seed(123)
    # n = 3
    # population_size = 10
    #
    # # generate the sudoku board
    # solved_board = generate_sudoku(n)
    # numpy.array(solved_board)
    #
    # # generate the puzzle board and the map of where we removed values
    # puzzle_board, removed_map = generate_removal_map(solved_board)

    puzzle_board = [[0,6,0,2,0,9,0,0,0],[0,0,0,0,3,0,0,1,0],[1,0,0,6,0,0,0,0,9],[4,2,0,5,0,0,0,9,0],[0,0,5,3,0,2,8,6,0],[0,8,3,1,0,0,0,2,4],[8,7,0,9,0,6,0,3,5],[3,4,0,0,5,0,2,7,0],[2,0,6,0,7,3,0,0,1]]
    print("Sudoku Question:")
    pretty_print(puzzle_board)
    puzzle_board_question = np.array(puzzle_board).reshape((9,9)).astype(int)

    #Solve using Genetic algorithm
    s = gss.Sudoku()
    s.load(puzzle_board_question)
    start_time_GA = time.time()
    generation, solution, progress = s.solve()
    if (solution):
        if generation == -1:
            print("Invalid inputs, Please check puzzle board")
        elif generation == -2:
            print("No solution found, Please try again")
        else:
            board_Solution = solution.values
            time_elapsed_GA = '{0:6.2f}'.format(time.time() - start_time_GA)
            str_print_GA = "Genetic Algorithm Solution found at generation: " + str(generation) + \
                        "\n" + "Time elapsed: " + str(time_elapsed_GA) + "s"
            print(str_print_GA)
            pretty_print(board_Solution)

    #Solve using Backtracking algorithm
    start_time_BT = time.time()
    solution_BT = btss.solve_sudoku(puzzle_board)
    print("Execution Started: Backtracking")
    if (solution_BT):
        time_elapsed_BT = '{0:6.2f}'.format(time.time() - start_time_BT)
        str_print_BT = "Backtracking Solution found:\n" + "Time elapsed: " + str(time_elapsed_BT) + "s"
        print(str_print_BT)
        pretty_print(puzzle_board)
    else:
        print("Invalid Puzzle board! Unable to solve puzzle")



