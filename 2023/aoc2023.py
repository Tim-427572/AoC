import hashlib
# import sys
# import math
# import time
# import copy
# import curses # pip install windows-curses
import inspect
import pickle
import random
import re
import socket
from os import path

# import functools
# import itertools
# import statistics
# import collections
import numpy as np
import pyglet
# import string
import requests

# Advent of Code
# Never did spend the time to work out how to get oAuth to work so this code expects you to
# manually copy over your session cookie value.
# Using a web browser inspect the cookies when logged into the Advent of Code website.
# Copy the value from the "session" cookie into a text file called "session.txt"

# Constants
_code_path = r'c:\AoC'
_offline = False
_year = 2023


def _check_internet(host="8.8.8.8", port=53, timeout=2):
    """
    Attempt to check for the firewall by connecting to Google's DNS.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    # except socket.error as ex:
        # print(ex)
    except socket.error:
        return False


def _pull_puzzle_input(day, seperator, cast):
    """
    Pull the puzzle data from the AOC website.

    :param day: (int,str) the AoC day puzzle input to fetch or an example puzzle string
    :param seperator: (str,None) A string separator to pass into str.split when consuming the puzzle data.
        If None or "" don't try and split the puzzle input.
    :param cast: (None,type) A Python function often a type cast (int, str, lambda) to be run against each data element.

    :return: tuple of the data.
    """
    global _work, _offline, _code_path

    if _offline:
        with open(_code_path + r"\{}\day{}.txt".format(_year, day)) as file_handler:
            data_list = file_handler.read().split(seperator)
    elif isinstance(day, str):  # An example string
        data_list = day.split(seperator)
    else:
        if not path.exists(_code_path + "/session.txt"):
            raise Exception("Using the web browser get the session cookie value\nand put it as a string in {}".format(_code_path + "\session.txt"))  # noqa: W605
        with open(_code_path + "/session.txt", 'r') as session_file:
            session = session_file.read()
        # Check to see if behind the firewall.
        if _check_internet():
            proxy_dict = {}
        else:
            proxy_dict = {'http': 'proxy-dmz.intel.com:911',
                          'https': 'proxy-dmz.intel.com:912'}
        header = {'Cookie': 'session={:s}'.format(session.rstrip('\n'))}
        with requests.Session() as session:
            resp = session.get('https://adventofcode.com/{}/day/{}/input'.format(_year, day), headers = header, proxies = proxy_dict)  # noqa: E251
            _ = resp.text.strip("\n")
            if resp.ok:
                if seperator in [None, ""]:
                    data_list = [resp.text]
                else:
                    data_list = resp.text.split(seperator)
            else:
                print("Warning website error")
                return ()

    if data_list[-1] == "":
        data_list.pop(-1)
    if cast is not None:
        data_list = [cast(x) for x in data_list]
    return tuple(data_list)


# Cache the data in a pickle file.
def get_input(day, seperator, cast, override=False):
    """
    Helper function for the daily puzzle information.
    If the puzzle data does not exist it attempts to pull it from the website.
    Caches the puzzle data into a pickle file so that re-runs don't have the performance
    penalty of fetching from the Advent Of Code website.
    :param day: (int, str) the AoC day puzzle input to fetch or a string of the puzzle example.
    :param seperator: (str) A string separator to pass into str.split when consuming the puzzle data.
    :param cast: (None,type) A Python function often a type cast (int, str, lambda) to be run against each data element.
                             None - do not apply a function/cast to the data.
    :param override: (bool) True = Fetch the data again instead of using the cached copy.

    :return: tuple containing the puzzle data
    """
    global _code_path
    if path.exists(_code_path + r'\{}\input.p'.format(_year)):
        puzzle_dict = pickle.load(open(_code_path + r'\{}\input.p'.format(_year), 'rb'))
    else:  # No pickle file, will need to make a new one.
        puzzle_dict = {}

    puzzle_input = puzzle_dict.get(day, None)

    if puzzle_input is None or override is True:
        puzzle_input = _pull_puzzle_input(day, seperator, cast)
        if isinstance(day, int):  # only save the full puzzle data to the pickle file.
            puzzle_dict[day] = puzzle_input
            pickle.dump(puzzle_dict, open(_code_path + r'\{}\input.p'.format(_year), 'wb'))
    return puzzle_input


class Viz(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_draw(self):
        self.clear()
        label = pyglet.text.Label("Hello, world",font_name='Times New Roman',font_size=36, x=self.width//2, y=self.height//2, anchor_x='center',anchor_y='center')
        label.draw()

def v():
    #label = pyglet.text.Label("Hello, world",font_name='Times New Roman',font_size=36, anchor_x='center',anchor_y='center')
    _ = Viz(512, 512, "Test",resizable=False)
    pyglet.app.run()


class Point:
    """
    A point object to use with 2D arrays where y/row is the first index and x/column is the second.
    """
    def __init__(self, y, x, shape=None, empty_shape="."):
        self.x = x
        self.y = y
        self.contains = shape
        self.is_empty = True if self.contains == empty_shape else False
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.up_right = None
        self.up_left = None
        self.down_right = None
        self.down_left = None
    def move(self, direction, steps=1):
        if direction in ["u", "n", "up", "north"]:
            self.y -= steps
        if direction in ["d", "s", "down", "south"]:
            self.y += steps
        if direction in ["r", "e", "right", "east"]:
            self.x += steps
        if direction in ["l", "w", "left", "west"]:
            self.x -= steps
        if direction in ["ur", "ne", "up_right", "north_east"]:
            self.y -= steps
            self.x += steps
        if direction in ["ul", "nw", "up_left", "north_west"]:
            self.y -= steps
            self.x -= steps
        if direction in ["dr", "se", "down_right", "south_east"]:
            self.y += steps
            self.x += steps
        if direction in ["dl", "sw", "down_left", "south_west"]:
            self.y += steps
            self.x -= steps
    def position(self):
        return (self.y, self.x)
    def p(self):
        return (self.y, self.x)
    def show(self):
        print(f"({self.y},{self.x}) {self.contains}")
        if self.up:
            print(f"up: ({self.up[0].y}, {self.up[0].x}) {self.up[1]}")
        if self.down:
            print(f"down: ({self.down[0].y}, {self.down[0].x}) {self.down[1]}")
        if self.left:
            print(f"left: ({self.left[0].y}, {self.left[0].x}) {self.left[1]}")
        if self.right:
            print(f"right: ({self.right[0].y}, {self.right[0].x}) {self.right[1]}")


def dfs(graph, node):  # Example function for DFS
    visited = set()
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor)


def bfs(graph, node):  # Example function for BFS
  visited = set()
  queue = [node]

  while queue:          # Creating loop to visit each node
    this_node = queue.pop(0) 
    print(this_node)

    for neighbor in graph[this_node]:
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append(neighbor)


def day1(example=False):
    """
    So it begins!
    """
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    if example == 1:
        day = ("1abc2\n"
               "pqr3stu8vwx\n"
               "a1b2c3d4e5f\n"
               "treb7uchet\n")
    if example == 2:
        day = ("two1nine\n"
               "eightwothree\n"
               "abcone2threexyz\n"
               "xtwone3four\n"
               "4nineeightseven2\n"
               "zoneight234\n"
               "7pqrstsixteen\n")

    calibration_doc = get_input(day, '\n', None, False)
    p1_calibration = 0
    p2_calibration = 0
    numbers = "1234567890"
    number_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]   
    for new_value in calibration_doc:
        if any(digit in new_value for digit in numbers):  # Conditional just for the part 2 example, not needed for the real puzzle input.
            digit_index_list = [f(x) for x in numbers if x in new_value for f in (lambda x: new_value.index(x), lambda x: new_value.rindex(x))]
            value = f"{new_value[min(digit_index_list)]}{new_value[max(digit_index_list)]}"
        else:
            value = "00"
        p1_calibration += int(value)
        if any(word in new_value for word in number_words):  # Skip if there are no number words in the calibration.
            first_word_dict = dict([(new_value.index(x), numbers[number_words.index(x)]) for x in number_words if x in new_value])
            value = first_word_dict[min(first_word_dict.keys())] + value[1] if min(first_word_dict.keys()) < min(digit_index_list) else value
            last_word_dict = dict([(new_value.rindex(x), numbers[number_words.index(x)]) for x in number_words if x in new_value])
            value = value[0] + last_word_dict[max(last_word_dict.keys())] if max(last_word_dict.keys()) > max(digit_index_list) else value
        p2_calibration += int(value)
    print(f"Part 1 the sum of calibration values is {p1_calibration}")
    print(f"Part 2 the sum of calibration values is {p2_calibration}")


def day2(example=False, reload=False):
    if example:
        day = ("Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green\n"
               "Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue\n"
               "Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red\n"
               "Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red\n"
               "Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))

    puzzle = get_input(day, "\n", None, reload)
    p1_cubes = {"red":12, "green":13, "blue":14} 
    p1_possible_sum = p2_power_sum = 0
    for game in puzzle:
        game_id, game_info = game.split(":")
        game_id = game_id.split(" ")[1]
        possible_game = True
        min_cubes = {"red":0, "green":0, "blue":0} 
        for subset in game_info.split(";"):
            for cubes in subset.split(","):
                number, color = cubes.strip().split(" ")
                number = int(number)
                possible_game = False if p1_cubes[color] < number else possible_game
                min_cubes[color] = number if min_cubes[color] < number else min_cubes[color]
        if possible_game:
            p1_possible_sum += int(game_id)
        p2_power_sum += np.prod(list(min_cubes.values()))
    print(f"Part 1 sum of possible game IDs is {p1_possible_sum}")
    print(f"Part 2 sum of game cube power is {p2_power_sum}")


def day3(example=False, reload=False):
    if example:
        day = ("467..114..\n"
               "...*......\n"
               "..35..633.\n"
               "......#...\n"
               "617*......\n"
               ".....+.58.\n"
               "..592.....\n"
               "......755.\n"
               "...$.*....\n"
               ".664.598..\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    schematic = get_input(day, "\n", lambda x: "." + x + ".", None)
    schematic = ["."*len(schematic[0])] + list(schematic) + ["."*len(schematic[0])] 
    schematic_array = np.array(list(map(list, schematic)))
    number_positions = number_hashes = {}
    p1_sum = p2_sum = 0
    num_regex = re.compile("[0-9]+")
    for row_idx, row in enumerate(schematic): 
        numbers = num_regex.finditer(row)
        for number in numbers:
            num_hash = random.getrandbits(128)
            number_hashes[num_hash] = int(number.group())
            for i in range(*number.span()):
                number_positions[(row_idx, i)] = num_hash
    p1_adj_numbers = set()
    for symbol_r, symbol_c in zip(*np.where(~np.char.equal(schematic_array, ".") & ~np.char.isdigit(schematic_array))):
        p2_adj_gears = set()
        for row in range(symbol_r - 1, symbol_r + 2):
            for col in range(symbol_c - 1, symbol_c + 2):
                if (row, col) in number_positions.keys():
                    p1_adj_numbers.add(number_positions[(row, col)])
                    p2_adj_gears.add(number_positions[(row, col)])
        if len(p2_adj_gears) == 2:
            p2_sum += np.prod(list(map(lambda x: number_hashes[x], p2_adj_gears)))
    p1_sum = sum(list(map(lambda x: number_hashes[x], p1_adj_numbers)))

    print(f"Part 1 the sum of engine part numbers is {p1_sum}")
    print(f"Part 2 the sum of engine gears ratio {p2_sum}")


def day4(example=False, reload=False):
    if example:
        day = ("Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53\n"
               "Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19\n"
               "Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1\n"
               "Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83\n"
               "Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36\n"
               "Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    cards = {}
    score = 0
    for card in puzzle:
        card_number, values = card.split(":")
        winning, have = values.split("|")
        card_number = int(card_number.split()[1])
        win_set = set(map(int, winning.split()))
        have_set = set(map(int, have.split()))
        matches = len(win_set.intersection(have_set))
        if matches:
            score += 2**(matches-1)
        cards[card_number] = matches
    print(f"Part 1 the cards are worth {score}")
    orig_card_list = list(cards.keys())
    final_list = []
    while orig_card_list:
        card = orig_card_list.pop()
        final_list.append(card)
        if cards[card]:
           orig_card_list += list(range(card + 1, (card + cards[card] + 1)))
    print(f"Part 2 the total number of scratchcards is {len(final_list)}")


def day5(example=False, reload=False):
    if example:
        day = """
"""
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    print(day)
    return
    puzzle = get_input(day, "\n", None, reload)
    p1 = p2 = 0
    print(p1)
    print(p2)


def go(day=1):
    try:
        return eval("day{}".format(day))
    except Exception as e:
        print(e)

