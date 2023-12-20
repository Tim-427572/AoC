import hashlib
# import sys
# import math
# import time
import copy
# import curses # pip install windows-curses
import inspect
import pickle
import random
import re
import socket
from os import path

# import functools
import itertools
# import statistics
import collections
import numpy as np
import pyglet
# import string
import requests
import functools
import heapq

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


def _pull_puzzle_input(day, seperator, cast=None):
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
def get_input(day, seperator, cast=None, override=False):
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


def get_np_input(day, seperator, cast=None, splitter=None, dtype=None, override=False):
    """
    Wrap get_input and cast the allow casting the data type too.
    returns a numpy array instead of the tuple array that get_input does.
    """
    day_input = get_input(day, seperator, cast, override)
    if splitter is None:
        return np.array(day_input, dtype=dtype)
    else:
        temp = []
        for r in day_input:
            foo = splitter(r)
            #print(foo)
            temp.append(foo)
            #temp.append(splitter(r))
        return np.array(temp, dtype=dtype)


def print_np(array):
    """
    Small script to print a numpy array to the console visually similar to the puzzles in AoC.
    """
    if array.dtype == np.dtype("<U1"):
        for row in array:
            print("".join(row))
    else:
        for row in array:
            print(np.array2string(row, separator="", max_line_width=600)[1:-1])


"""
# Some code experiments in a visualization module instead of using curses.
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
"""

class Point_Object:
    """
    A point object to use with 2D arrays where y/row is the first index and x/column is the second.
    Useful when you want to turn the 2D map into a graph.
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


# A thing that isn't really a tuple which makes tracking 2D points easier.
class Coordinate(tuple):  # noqa: SLOT001
    """
    Like a tuple but not.

    Used to store 2D position but still allow hashing and (x,y) notation which I like.
    """

    def __mul__(self, other):
        """Multiply a scaler with this coordinate."""
        return Coordinate(x * other for x in self)
    def __neg__(self):
        """Turn the coordinate negative."""
        return Coordinate(-1 * x for x in self)
    def __add__(self, other):
        """Add two coordinates or a coordinate and a tuple."""
        return Coordinate(x + y for x, y in zip(self, other))
    def __lt__(self, other):
        """Use to test if coordinate in 2D array."""
        return all(x < y for x, y in zip(self, other))
    def __le__(self, other):
        """Use to test if coordinate in 2D array."""
        return all(x <= y for x, y in zip(self, other))
    def __gt__(self, other):
        """Use to test if coordinate in 2D array."""
        return all(x > y for x, y, in zip(self, other))
    def __ge__(self, other):
        """Use to test if coordinate in 2D array."""
        return all(x >= y for x, y, in zip(self, other))
    def __setitem__(self, key, value):
        """Ok, look it really isn't a tuple."""
        self_list = list(self)
        self_list[key] = value
        # print(l)
        return Coordinate(tuple(self_list))
    def manhattan_dist(self, other):
        """Calculate the manhattan distance between this coordinate and another."""
        return abs(self[0] - other[0]) + abs(self[1] - other[1])


# Dictionary to make walking the 2D maps easier.
move_dict = {"u": (-1, 0), "n": (-1, 0), "up": (-1, 0), "north": (-1, 0),
             "d": (1, 0), "s": (1, 0), "down": (1, 0), "south": (1, 0),
             "r": (0, 1), "e": (0, 1), "right": (0, 1), "east": (0, 1),
             "l": (0, -1), "w": (0, -1), "left": (0, -1), "west": (0, -1),
             "ur": (-1, 1), "ne": (-1, 1), "up-right": (-1, 1), "north-east": (-1, 1),
             "dr": (1, 1), "se": (1, 1), "down-right": (1, 1), "south-east": (1, 1),
             "ul": (-1, -1), "nw": (-1, -1), "up-left": (-1, -1), "north-west": (-1, -1),
             "dl": (1, -1), "sw": (1, -1), "down-left": (1, -1), "south-west": (1, -1)}

_right = {"r": "d", "e": "s", "right": "down", "east": "south",
          "l": "u", "w": "n", "left": "up", "west": "north",
          "u": "r", "n": "e", "up": "right", "north": "east",
          "d": "l", "s": "w", "down": "left", "south": "west"}

_left = {"r": "u", "e": "n", "right": "up", "east": "north",
         "l": "d", "w": "s", "left": "down", "west": "south",
          "u": "l", "n": "w", "up": "left", "north": "west",
          "d": "r", "s": "e", "down": "right", "south": "east"}

turn_dict = {"r": _right, "right": _right, "cw": _right, "clockwise": _right,
             "l": _left, "left": _left, "ccw": _left, "counterclockwise": _left}


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
    """
    Minimum cube set for the games.
    """
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
    """
    Finding numbers adjacent to things, then finding pairs which touch the * character.
    """
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
    """
    Lottery cards!
    """
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


def _day5_fwd(seed, ranges):
    for dest, src, size in ranges:
        if src <= seed < src + size:
            return seed + dest - src
    return seed

def _day5_range(ranges, this_map):
    result = []
    for dest, src, size in this_map:
        src_end = src + size
        new_ranges = []
        # Look for ranges that overlap this mapping section
        # If it does overlap, remap the range based on the rules.
        while ranges:
            start, end = ranges.pop()
            left = (start, min(end, src))
            inside = (max(start, src), min(src_end, end))
            right = (max(src_end, start), end)
            if left[1] > left[0]:
                new_ranges.append(left)
            if inside[1] > inside[0]:
                result.append((inside[0] - src + dest, inside[1] - src + dest))
            if right[1] > right[0]:
                new_ranges.append(right)
        ranges = new_ranges
    return result + ranges


def day5(example=False, reload=False):
    """
    Translate seeds into locations. Then translate ranges of seeds into ranges of locations.
    """
    if example:
        day = ("seeds: 79 14 55 13\n"
               "\n"
               "seed-to-soil map:\n"
               "50 98 2\n"
               "52 50 48\n"
               "\n"
               "soil-to-fertilizer map:\n"
               "0 15 37\n"
               "37 52 2\n"
               "39 0 15\n"
               "\n"
               "fertilizer-to-water map:\n"
               "49 53 8\n"
               "0 11 42\n"
               "42 0 7\n"
               "57 7 4\n"
               "\n"
               "water-to-light map:\n"
               "88 18 7\n"
               "18 25 70\n"
               "\n"
               "light-to-temperature map:\n"
               "45 77 23\n"
               "81 45 19\n"
               "68 64 13\n"
               "\n"
               "temperature-to-humidity map:\n"
               "0 69 1\n"
               "1 0 69\n"
               "\n"
               "humidity-to-location map:\n"
               "60 56 37\n"
               "56 93 4\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    maps = []
    ranges = []
    i = 0
    while i < len(puzzle):
        line = puzzle[i]
        if "seeds" in line:
            seeds = list(map(int, (line.split(":")[1].split())))
            i += 1
        elif line == "":
            if ranges:
                ranges.sort()
                maps.append(ranges.copy())
                ranges = []
            i+=1
        elif ":" in line: # one of the map names.
            i+=1
        else:
            dest, src, size = map(int, line.split())
            ranges.append((dest, src, size))
            i+=1
    if ranges:  # Final range at the end of the input.
        ranges.sort()
        maps.append(ranges.copy())

    p1_min = None
    for seed in seeds:
        for ranges in maps:
            seed = _day5_fwd(seed, ranges)
        p1_min = seed if( p1_min is None or seed < p1_min) else p1_min
    print(f"Part 1 the minimum location is {p1_min}")

    p2_min = None
    seed_ranges = list(zip(*[iter(seeds)]*2))
    # Walk the seed ranges
    for start, size in seed_ranges:
        ranges = [(start, start+size)]
        for this_map in maps:
            ranges = _day5_range(ranges, this_map)
        range_min = min(ranges)[0]
        p2_min = range_min if p2_min is None or range_min < p2_min else p2_min
    print(f"Part 2 the minimum location is {p2_min}")


def day6(example=False, reload=False):
    """
    Boat races! Didn't bother trying to do the actual math, brute force FTW.
    """
    if example:
        day = ("Time:      7 15 30\n"
               "Distance:  9 40 200\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    p1_answer = p2_answer = 1
    times = list(map(int, puzzle[0].split(":")[1].split()))
    distances = list(map(int, puzzle[1].split(":")[1].split()))
    times.append(int(puzzle[0].split(":")[1].replace(" ", "")))
    distances.append(int(puzzle[1].split(":")[1].replace(" ","")))
    for race_time, distance in zip(times, distances):
        ways = 0
        for hold in range(race_time):
            if hold * (race_time - hold) > distance:
                ways += 1
            elif ways != 0:
                break
        if race_time == times[-1]:
            p2_answer *= ways
        else:
            p1_answer *= ways
    print(f"The part 1 answer is {p1_answer}")
    print(f"The part 2 answer is {p2_answer}")


def _day7_check_type(hand):
    """
    Helper function which takes a camel card hand and returns a string describing its type.
    """
    hand_decode = {(5,1):"five of a kind",
                   (4,1):"four of a kind",
                   (3,2):"full house",
                   (3,1):"three of a kind",
                   (2,2):"two pair",
                   (2,1):"one pair",
                   (1,1):"high card"}
    counts = collections.Counter(hand + "!").most_common(2) # Pad with ! to handle the five of a kind case.
    return hand_decode[list(zip(*counts))[1]]


def _day7_hand_sort(left, right):
    """
    Custom sorting function for the strength of the hands
    going to use 1 for the J when scoring part instead of having two of these functions.
    """
    card_value = {"1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
    for left_char, right_char in zip(left, right):
        if card_value[left_char] > card_value[right_char]:
            return 1
        if card_value[left_char] < card_value[right_char]:
            return -1
    return 0


def day7(example=False, reload=False):
    """
    Scoring camel card hands, used a custom sort function to make the code easier to read.
    """
    if example:
        day = ("32T3K 765\n"
               "T55J5 684\n"
               "KK677 28\n"
               "KTJJT 220\n"
               "QQQJA 483\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    type_order = ["high card", "one pair", "two pair", "three of a kind", "full house", "four of a kind", "five of a kind"]
    p1_answer = p2_answer = 0
    p1_hands_of_each_type = collections.defaultdict(list)
    p2_hands_of_each_type = collections.defaultdict(list)
    hand_bid = {}
    for line in puzzle:
        hand = line.split()[0]
        score = int(line.split()[1])
        hand_bid[hand] = score
        p1_hands_of_each_type[_day7_check_type(hand)].append(hand)
        # Replace the J wild cards to match the most common other card.
        if "J" in hand and hand.count("J") < 5:
            new_hand = hand.replace("J", collections.Counter(hand.replace("J", "")).most_common(1)[0][0])
        else:
            new_hand = hand
        p2_hands_of_each_type[_day7_check_type(new_hand)].append(hand.replace("J", "1"))  # Replace J with 1 so I don't need two sorting functions.
    p1_strengths = []
    p2_strengths = []
    for hand_type in type_order:
        p1_strengths += sorted(p1_hands_of_each_type[hand_type], key=functools.cmp_to_key(_day7_hand_sort))
        p2_strengths += sorted(p2_hands_of_each_type[hand_type], key=functools.cmp_to_key(_day7_hand_sort))

    # Ok we have the hands in order, time to score them.
    rank = 1
    for p1_hand, p2_hand in zip(p1_strengths, p2_strengths):
        p1_answer += rank * hand_bid[p1_hand]
        p2_answer += rank * hand_bid[p2_hand.replace("1", "J")]  # Put the J back so we can look up the score.
        rank += 1
    
    print(f"The part 1 total winnings is {p1_answer}")
    print(f"The part 2 total winnings is {p2_answer}")


def day8(example=False, reload=False):
    """
    Moving from A to Z.
    Used LCM in part 2 but made the assumption that the cycle period from *A to *Z is the same as from *Z back to *z
    """
    if example:
        day = ("LR\n"
                "\n"
                "11A = (11B, XXX)\n"
                "11B = (XXX, 11Z)\n"
                "11Z = (11B, XXX)\n"
                "22A = (22B, XXX)\n"
                "22B = (22C, 22C)\n"
                "22C = (22Z, 22Z)\n"
                "22Z = (22B, 22B)\n"
                "XXX = (XXX, XXX)\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    instructions = list(map(int, puzzle[0].replace("R","1").replace("L","0")))
    instructions = itertools.cycle(instructions)
    puzzle = puzzle[2:]
    p1_answer = 0
    nodes = {}
    a_elements = []
    step_list = []
    for line in puzzle:
        this_element, left, right = re.findall("[0-9A-Z]+", line)
        nodes[this_element] = (left, right)
        if this_element[-1] == "A":
            a_elements.append(this_element)
    for element in a_elements:
        steps = 0
        while True:
            element = nodes[element][next(instructions)]
            steps += 1
            p1_answer = steps if element == "ZZZ" else p1_answer
            if element[-1] == "Z":
                step_list.append(steps)
                break
    print(f"The totals steps for part 1 is {p1_answer}")
    print(f"The part 2 answer is {np.lcm.reduce(step_list, dtype='int64')}")


def day9(example=False, reload=False):
    """
    Weird subtraction puzzle.
    """
    if example:
        day = ("0 3 6 9 12 15\n"
               "1 3 6 10 15 21\n"
               "10 13 16 21 30 45\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    p1_answer = []
    p2_answer = []
    value_histories = []
    for line in puzzle:
        value_histories.append([list(map(int, line.split()))])
    # Fill in each value history
    for value_history in value_histories:
        while True:
            value_history.append(np.diff(value_history[-1]))
            if all(x ==0 for x in value_history[-1]):
                break
    for value_history in value_histories:
        p1 = p2 = 0
        for i in range(len(value_history) - 1, -1, -1):
            p1=value_history[i][0] + p1
            p2=value_history[i][0] - p2
        p1_answer.append(p1)
        p2_answer.append(p2)
    print(f"Part 1 sum is {sum(p1_answer)}")
    print(f"Part 2 sum is {sum(p2_answer)}")


def d10_bfs(field, loop, coordinate, dist):
    """
    A BFS search? realy only takes the two paths in opposite directions around the loop.
    """
    global move_dict
    queue = [(coordinate, dist)]
    """
    """
    while queue:
        this_node, this_distance = queue.pop(0)
        loop[this_node] = this_distance
        for direction, here, there in [("up","|SJL","S|7F"),
                                       ("down","S|7F","S|LJ"),
                                       ("left","S-J7","S-FL"),
                                       ("right","S-FL","S-J7")]:
            neighbor = this_node + move_dict[direction]
            #print(this_node, neighbor)
            if (neighbor not in loop and
                field[this_node] in here and
                field[neighbor] in there):
                queue.append((neighbor, this_distance + 1))


def day10(example=False, reload=False):
    """
    I am not 100% sure the P2 algorithm would work on any input but it works on the examples and my input which is good enough.
    """
    if example == 1:
        day = ("-L|F7\n"
               "7S-7|\n"
               "L|7||\n"
               "-L-J|\n"
               "L|-JF\n")
    elif example == 2:
        day = ("...........\n"
               ".S-------7.\n"
               ".|F-----7|.\n"
               ".||.....||.\n"
               ".||.....||.\n"
               ".|L-7.F-J|.\n"
               ".|..|.|..|.\n"
               ".L--J.L--J.\n"
               "...........\n")
    elif example == 3:
        day = (".F----7F7F7F7F-7....\n"
               ".|F--7||||||||FJ....\n"
               ".||.FJ||||||||L7....\n"
               "FJL7L7LJLJ||LJ.L-7..\n"
               "L--J.L7...LJS7F-7L7.\n"
               "....F-J..F7FJ|L7L7L7\n"
               "....L7.F7||L7|.L7L7|\n"
               ".....|FJLJ|FJ|F7|.LJ\n"
               "....FJL-7.||.||||...\n"
               "....L---J.LJ.LJLJ...\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    field = get_np_input(day, "\n", splitter=list, dtype=str, override=reload)
    field = np.pad(field, 1, mode="constant", constant_values=".")
    #print_np(field)
    start = Coordinate(np.argwhere(field == "S")[0])
    loop = {}
    d10_bfs(field, loop, start, 0)
    print(f"Part 1: it is {max(loop.values())} steps to the farthest point in the loop")
    loop_only = np.full(field.shape, fill_value = ".", dtype=str)
    inside_points = []
    for point in loop:
        loop_only[point] = field[point]
    for y, row in enumerate(loop_only):
        for x, col in enumerate(row):
            if (y, x) in loop:
                continue
            crossings = np.count_nonzero(np.in1d(row[:x], ["S","J","L","|"]))
            if crossings % 2 != 0:
                inside_points.append((y,x))
                loop_only[y,x] = "I"
    # print_np(loop_only)
    print(f"Part 2 the number of points inside the loop is {len(inside_points)}")



def day11(universe_expansion = 2, example=False, reload=False,):
    """
    Mapping the universe.
    """
    if example:
        day = ("...#......\n"
               ".......#..\n"
               "#.........\n"
               "..........\n"
               "......#...\n"
               ".#........\n"
               ".........#\n"
               "..........\n"
               ".......#..\n"
               "#...#.....\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    universe = get_np_input(day, "\n", splitter=list, dtype=str, override=reload)
    empty_rows = np.where(~(universe == "#").any(axis=1))[0].tolist()
    empty_columns = np.where(~(universe == "#").any(axis=0))[0].tolist()
    galaxies = np.argwhere(universe == "#").tolist()
    universe_expansion -= 1  # Accound for the one existing empty line in the universe map.
    answer = 0
    for p1, p2 in itertools.combinations(galaxies, 2):
        row_range = range(*sorted([p1[0], p2[0]]))
        row_delta = len(row_range)
        row_delta += sum([universe_expansion for row in empty_rows if row in row_range])
        col_range = range(*sorted([p1[1], p2[1]]))
        col_delta = len(col_range)
        row_delta += sum([universe_expansion for col in empty_columns if col in col_range])
        # print(pair, (row_delta + col_delta))  # Was helping Max debug.
        answer += (row_delta + col_delta)
    print(f"The sum of the distances between galaxies is {answer}")


def day12(example=False, reload=False,):
    """
    """
    if example:
        day = ("???.### 1,1,3\n"
               ".??..??...?##. 1,1,3\n"
               "?#?#?#?#?#?#?#? 1,3,1,6\n"
               "????.#...#... 4,1,1\n"
               "????.######..#####. 1,6,5\n"
               "?###???????? 3,2,1\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    p = get_input(day, "\n", cast=None, override=reload)
    #n = get_np_input(day, "\n", splitter=list, dtype=str, override=reload)
    p1 = p2 = 0
    for l in p:
        #print(l)
        sol_set = set()
        test_set = set()
        m,a = l.split(" ")
        a = list(map(int, a.split(",")))
        for r in itertools.product([".","#"], repeat=m.count("?")):
            t=m
            for i in r:
                t= t[:t.index("?")] + i + t[t.index("?")+1:]
            if t in test_set:
                continue
            test_set.add(t)
            s = re.findall("[#]+", t)
            #print(t,s)
            #input()
            if len(s) != len(a):
                continue
            si = [len(x) for x in s]
            if si == a:
                sol_set.add(t)
        #print(sol_set)
        print(m, a, len(sol_set))
        p1 += len(sol_set)
        #input()
            
    print(f"p1 {p1}")
    print(f"p2 {p2}")


# Memoization FTW!
@functools.lru_cache(maxsize=None)
def day12_recursion(s, in_a_spring, sizes):
    if s == "":  # end of the string.
        if in_a_spring is None and len(sizes) == 0: # Not in a spring and nothing left in the sizes list, success
            return 1
        elif in_a_spring is not None and len(sizes) == 1 and in_a_spring == sizes[0]:  # Ending on a spring and the size matches.
            return 1
        else:
            return 0
    possible_springs = s.count("#") + s.count("?")
    this_chr = s[0]
    remaining_str = s[1:]
    # Attempt early abort to increase speed.
    if in_a_spring is None:
        if possible_springs < sum(sizes): # Not enough room.
            return 0
    else:
        if (len(sizes) == 0 or  # Abort if we are in a spring but there are no more sizes.
            possible_springs + in_a_spring < sum(sizes) or  # Not enough characters which could be # left.
            this_chr == "." and in_a_spring != sizes[0]):  # This is the end of a spring section but the size does not match.
            return 0
    remaining_sizes = sizes[1:]
    possible_arrangements = 0
    if in_a_spring is None:  # In between springs
        if this_chr in ["#","?"]:  # Starting a new spring.
            possible_arrangements += day12_recursion(remaining_str, 1, sizes)
        if this_chr in ["?","."]:  # Search for the next spring.
            possible_arrangements += day12_recursion(remaining_str, None, sizes)    
    else:  # Inside the run of a spring.
        if this_chr == ".":  # End of this spring, start the next search.
            possible_arrangements += day12_recursion(remaining_str, None, remaining_sizes)
        if this_chr == "?" and in_a_spring == sizes[0]: # End of this spring because the size matches, start the next.
            possible_arrangements += day12_recursion(remaining_str, None, remaining_sizes)
        if this_chr in ["#","?"]:  # Continue along this spring.
            possible_arrangements += day12_recursion(remaining_str, in_a_spring+1, sizes)

    return possible_arrangements


def day12_p2(example=False, reload=False,):
    """
    """
    if example:
        day = ("???.### 1,1,3\n"
               ".??..??...?##. 1,1,3\n"
               "?#?#?#?#?#?#?#? 1,3,1,6\n"
               "????.#...#... 4,1,1\n"
               "????.######..#####. 1,6,5\n"
               "?###???????? 3,2,1\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    p = get_input(day, "\n", cast=None, override=reload)
    # n = get_np_input(day, "\n", splitter=list, dtype=str, override=reload)
    p2 = 0
    for l in p:
        s, sizes = l.split(" ")
        sizes = list(map(int, sizes.split(",")))
        sizes = tuple(sizes * 5)  # Need this to be hashable for memoization!
        unfolded = s
        for _ in range(4):
            unfolded += "?" + s
        p2 += day12_recursion(unfolded, None, sizes)
    print(p2)


def _day13_reflection(array):
    reflection_row = smudge_row = 0  # Using 0 makes the math work.
    # Walk the array rows and get equal sizes slices (remove anything that does not overlap)
    for row in range(1, array.shape[0]):
        top = array[:row]
        bottom = array[row:]
        if top.shape[0] < bottom.shape[0]:
            bottom = bottom[:top.shape[0]]
        else:
            top = top[bottom.shape[0] * -1:]
        # Subtract the top from the flipped bottom
        diff = top - np.flip(bottom, axis=0)
        if np.count_nonzero(diff) == 0:  # Exact match => symmetric
            reflection_row = row
        if np.count_nonzero(diff) == 1:  # Exactly one miss-match => smudge
            smudge_row = row
    return reflection_row, smudge_row


def day13(example=False, reload=False):
    """
    Find reflections.

    Uses numpy to find matching arrays.
    """
    if example:
        day = ("#.##..##.\n"
               "..#.##.#.\n"
               "##......#\n"
               "##......#\n"
               "..#.##.#.\n"
               "..##..##.\n"
               "#.#.##.#.\n"
               "\n"
               "#...##..#\n"
               "#....#..#\n"
               "..##..###\n"
               "#####.##.\n"
               "#####.##.\n"
               "..##..###\n"
               "#....#..#\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", cast=None, override=reload)
    temp = []
    maps = []
    to_int_dict = {".": 0, "#": 1}
    for line in puzzle:
        if not line:
            maps.append(np.array(temp))
            temp = []
        else:
            temp.append([to_int_dict[x] for x in list(line)])
    maps.append(np.array(temp))
    p1_answer = p2_answer = 0
    for this_map in maps:
        reflect_row, smudge_row = _day13_reflection(this_map)  # Check horizontal symmetry
        reflect_col, smudge_col = _day13_reflection(this_map.T)  # Check vertical symmetry
        # Add the stuff up, because the function returns 0 when it doesn't exist just blindly add.
        p1_answer += (reflect_col + (reflect_row * 100))
        p2_answer += (smudge_col + (smudge_row * 100))
    print(f"Part 1 answer: {p1_answer}")
    print(f"Part 2 answer: {p2_answer}")


@functools.lru_cache(maxsize=None)
def tilt_day14(round_tup, cube_tup, size, directions):
    global move_dict
    rock_set = set(round_tup)
    for direction in directions:
        moving = True
        while moving:
            moving = False
            new_rocks = set()
            while rock_set:
                prev_y, prev_x = rock_set.pop()
                while True:
                    new_y = prev_y + move_dict[direction][0]
                    new_x = prev_x + move_dict[direction][1]
                    new_pos = (new_y, new_x)
                    if (new_y < 0 or new_y >= size[0] or
                        new_x < 0 or new_x >= size[1] or
                        new_pos in cube_tup or
                        new_pos in new_rocks or
                        new_pos in rock_set):
                        new_y = prev_y
                        new_x = prev_x
                        break
                    prev_y = new_y
                    prev_x = new_x
                    moving = True
                new_rocks.add((new_y, new_x))
            rock_set = new_rocks
    return tuple(rock_set)


def day14(example=False, reload=False):
    """
    """
    if example:
        day = """O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#....
"""
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    p = get_input(day, "\n", cast=list, override=reload)
    # n = get_np_input(day, "\n", splitter=list, dtype=str, override=reload)
    rocks = []
    cubes = []
    size = (len(p), len(p[0]))
    for r,l in enumerate(p):
        for c,ch in enumerate(l):
            if ch == "O":
                rocks.append((r,c))
            elif ch == "#":
                cubes.append((r,c))
    rock_tup = tuple(rocks)
    cube_tup = tuple(cubes)
    cycle = {}
    # for spin in range(1, 5):
    for spin in range(1, 1000):
        rock_tup = tilt_day14(rock_tup, cube_tup, size, "nwse")

        """
        print("spin",spin)
        o = [list(".........."),list(".........."),list(".........."),list(".........."),list(".........."),list(".........."),list(".........."),list(".........."),list(".........."),list(".........."),]
        for r in rock_tup:
            o[r[0]][r[1]] = "O"
        for c in cube_tup:
            o[c[0]][c[1]] = "#"
        for r in o:
            print("".join(r))
        input()
        """

        p1 = 0
        for r in rock_tup:
            p1 += size[0] - r[0]
        # Found cycle and final result is evenly divisible by our current location.
        if rock_tup in cycle and (1000000000 - spin) % (spin - cycle[rock_tup]) == 0 and False:
            print(p1)
            break            
        cycle[rock_tup] = spin
        print("step",spin, "value", p1)


@functools.lru_cache(maxsize=None)
def hashit(string):
    """
    HASH - Holiday ASCII String Helper function.

    Day 15 lens name hashing function.
    """
    i = 0
    for char in string:
        i += ord(char)
        i = i * 17
        i = i % 256
    return i


def day15(example=False, reload=False):
    """Make a hash and put the lenses into the correct box."""
    if example:  # noqa: SIM108
        day = """rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7"""
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, ",", cast=lambda x: x.strip("\n"), override=reload)
    p1_answer = p2_answer = 0
    # Note: this solution only works in Python 3.7+ where the dictionary keys are insertion ordered.
    boxes = collections.defaultdict(dict)
    for line in puzzle:
        p1_answer += hashit(line)
        label, number = re.split("[=-]", line)
        box = hashit(label)
        if number:
            boxes[box][label] = int(number)
        else:
            boxes[box].pop(label, None)
    for box, labels in boxes.items():
        if labels:
            for index, (_, value) in enumerate(labels.items()):
                p2_answer += ((box + 1) * (index + 1) * value)
    print("Part 1", p1_answer)
    print("Part 2", p2_answer)


def d16_bfs(contraption, energized, beam, position_set):
    """
    Perform a BFS search.

    contraption - A numpy array containing the puzzle map.
    energized - A numpy array counting beams through each location.
    beam - tuple ((y,x), direction)
    """
    global move_dict, turn_dict  # noqa: PLW0602
    queue = [beam]
    while queue:
        this_pos, this_dir = queue.pop(0)
        if (this_pos, this_dir) not in position_set:
            position_set.add((this_pos, this_dir))
            next_pos = this_pos + move_dict[this_dir]
            #print(this_pos, this_dir, move_dict[this_dir], next_pos)
            #print_np(contraption)
            #print_np(energized)
            #print(this_pos, this_dir, queue)
            if (0, 0) <= next_pos < contraption.shape:  # Array bounds check
                energized[next_pos] += 1
                next_shape = contraption[next_pos]
                if this_dir in "rl" and next_shape == "\\":  # Turn right
                    queue.append((next_pos, turn_dict["right"][this_dir]))
                elif this_dir in "rl" and next_shape == "/":  # Turn left
                    queue.append((next_pos, turn_dict["left"][this_dir]))
                elif this_dir in "ud" and next_shape == "/":  # Turn right
                    queue.append((next_pos, turn_dict["right"][this_dir]))
                elif this_dir in "ud" and next_shape == "\\":  # Turn left
                    queue.append((next_pos, turn_dict["left"][this_dir]))
                elif this_dir in "rl" and next_shape == "|":  # Split
                    queue.append((next_pos, turn_dict["right"][this_dir]))
                    queue.append((next_pos, turn_dict["left"][this_dir]))
                elif this_dir in "ud" and next_shape == "-":  # Split
                    queue.append((next_pos, turn_dict["right"][this_dir]))
                    queue.append((next_pos, turn_dict["left"][this_dir]))
                else:
                    queue.append((next_pos, this_dir))
            #print(queue)
            #input()



def day16(example=False, reload=False):
    """Make floor wil be lava."""
    if example:  # noqa: SIM108
        day = (".|...\\....\n"
               "|.-.\\.....\n"
               ".....|-...\n"
               "........|.\n"
               "..........\n"
               ".........\\\n"
               "..../.\\\\..\n"
               ".-.-/..|..\n"
               ".|....-|.\\\n"
               "..//.|....\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    contraption = get_np_input(day, "\n", cast=None, splitter=list, dtype=str, override=reload)
    energized = np.zeros(contraption.shape, dtype=int)
    d16_bfs(contraption, energized, (Coordinate((0, -1)), "r"), set())
    print("Part 1", np.count_nonzero(energized))
    p2_answer = 0
    for d in "rl":
        for r in range(contraption.shape[0]):
            energized = np.zeros(contraption.shape, dtype=int)
            d16_bfs(contraption, energized, (Coordinate((r, -1)), d), set())
            e = np.count_nonzero(energized)
            if e > p2_answer:
                p2_answer = e
    for d in "ud":
        for r in range(contraption.shape[1]):
            energized = np.zeros(contraption.shape, dtype=int)
            d16_bfs(contraption, energized, (Coordinate((-1, r)), d), set())
            e = np.count_nonzero(energized)
            if e > p2_answer:
                p2_answer = e
    print("Part 2", p2_answer)


def day17_a_star(start, goal, min_chain, max_chain, n):
    """Perform an A* search for the path with the least heat loss."""
    queue = []
    dist = {start: 0}  # No heat loss.
    heapq.heappush(queue, (dist[start] + goal.man_dist(start[0]), start))

    while queue:  # noqa: RET503
        f_dist, cur = heapq.heappop(queue)
        if cur[0] == goal:
            return dist[cur]
        # Heuristic?
        if f_dist > dist[cur] + goal.man_dist(cur[0]):
            continue
        # Rules for valid neighbor positions.
        neighbors = []
        position, direction, length = cur
        for d in "nsew":
            if direction is not None:
                if d == turn_dict["left"][turn_dict["left"][direction]]:
                    continue
                if d == direction and length == max_chain:
                    continue
                if (d not in {direction, turn_dict["left"][turn_dict["left"][direction]]} and
                    length < min_chain):
                    continue
            new_pos = position + move_dict[d]
            if not ((0, 0) <= new_pos < n.shape):  # Array bounds check
                continue
            new_dir = d
            if new_dir != direction:  # noqa: SIM108
                new_chain = 1
            else:
                new_chain = length + 1
            neighbors.append((new_pos, new_dir, new_chain))
        # The search...
        for neighbor in neighbors:
            goal_dist = dist[cur] + n[neighbor[0]]
            if neighbor in dist and dist[neighbor] <= goal_dist:
                continue
            dist[neighbor] = goal_dist
            heapq.heappush(queue, (dist[neighbor] + goal.man_dist(neighbor[0]), neighbor))


def day17(example=False, reload=False):
    """Find the path of least heat loss."""
    if example:  # noqa: SIM108
        day = ("2413432311323\n"
               "3215453535623\n"
               "3255245654254\n"
               "3446585845452\n"
               "4546657867536\n"
               "1438598798454\n"
               "4457876987766\n"
               "3637877979653\n"
               "4654967986887\n"
               "4564679986453\n"
               "1224686865563\n"
               "2546548887735\n"
               "4322674655533\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    n = get_np_input(day, "\n", cast=None, splitter=list, dtype=int, override=reload)
    start = (Coordinate((0, 0)), None, 1)
    end = Coordinate(n.shape) + (-1, -1)  # noqa: RUF005
    print("Part 1", day17_a_star(start, end, 0, 3, n))
    print("Part 2", day17_a_star(start, end, 4, 10, n))


def day18(example=False, reload=False):
    """Calculate the size of the lagoon."""
    # Got really tired of shoelace algorithms and just pip installed a polygon package.
    from shapely.geometry import Polygon
    if example:  # noqa: SIM108
        day = ("R 6 (#70c710)\n"
               "D 5 (#0dc571)\n"
               "L 2 (#5713f0)\n"
               "D 2 (#d2c081)\n"
               "R 2 (#59c680)\n"
               "D 2 (#411b91)\n"
               "L 5 (#8ceee2)\n"
               "U 2 (#caa173)\n"
               "L 1 (#1b58a2)\n"
               "U 2 (#caa171)\n"
               "R 2 (#7807d2)\n"
               "U 3 (#a77fa3)\n"
               "L 2 (#015232)\n"
               "U 2 (#7a21e3)\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", cast=None, override=reload)
    p1_cur = Coordinate((0, 0))
    p2_cur = Coordinate((0, 0))
    p1_vertices = []
    p2_vertices = []
    p2_dir_decode = {"0": "r", "1": "d", "2": "l", "3": "u"}
    for line in puzzle:
        p1_dir, p1_dist, color = line.split(" ")
        p1_dist = int(p1_dist)
        p2_dir = p2_dir_decode[color[-2:-1]]
        p2_dist = color[2:-2]
        p2_dist = int(p2_dist, 16)
        p1_cur = p1_cur + (Coordinate(move_dict[p1_dir.lower()]) * p1_dist)
        p2_cur = p2_cur + (Coordinate(move_dict[p2_dir]) * p2_dist)
        p1_vertices.append(p1_cur)
        p2_vertices.append(p2_cur)
    # The digger creates a 1m wide trench, need to account for the 1/2m that is outside of the immmaginary polygon line.
    # Every exterior corner adds 1/4 a unit of the trench width. Every interior corner is subtracts a 1/4 unit.
    # Because this is a non-overlapping rectangular shape there are 4 more exertior corners than there are interior ones.
    # 4 * 1/4 = 1 additional unit of area needed.
    polygon = Polygon(p1_vertices)
    print(f"Part 1: The lagoon holds {int(polygon.length // 2) + int(polygon.area) + 1} cubic meters of lava")
    polygon = Polygon(p2_vertices)
    print(f"Part 2: The lagoon holds {int(polygon.length // 2) + int(polygon.area) + 1} cubic meters of lava")


def day19_bfs(workflows, rejected_paths, accepted_paths = []):
    """
    Find the path to R starting from in.

    Record the path (used for debug)
    Record the x,m,a,s value ranges that would result in taking this rejected path.
    """
    queue = [{"path": ["in"],
              "current":"in",
              "x":[1, 4001], "m":[1, 4001],
              "a":[1, 4001], "s":[1, 4001]}]
    while queue:
        this_node = queue.pop(0)
        else_node = copy.deepcopy(this_node)  # This node tracks the else condition for each instruction.
        for instruction in workflows[this_node["current"]]:
            if ":" in instruction:
                # This node starts with all the previous else conditions and then will add the next if condition.
                temp_node = copy.deepcopy(else_node)
                equation, next_inst = instruction.split(":")
                var, num = re.split("[<>]", equation)
                num = int(num)
                # Carefully adjusting the rangees of rejected parts so the format works with python range.
                if "<" in equation:
                    temp_node[var][1] = min(temp_node[var][1], num)
                    else_node[var][0] = max(else_node[var][0], num)
                else:  # it was >
                    temp_node[var][0] = max(temp_node[var][0], num + 1)
                    else_node[var][1] = min(else_node[var][1], num + 1)
                # Update this node for next instruction name.
                temp_node["current"] = next_inst
                temp_node["path"].append(next_inst)
                if temp_node["current"] == "R":  # Found R this is a complete rejected path
                    rejected_paths.append(temp_node)
                elif temp_node["current"] != "A":  # Haven't found R or A yet (drop the path if we get to A)
                    queue.append(temp_node)
                else:
                    accepted_paths.append(temp_node)
            else:  # This is the else condition, the next instruction we go to if all the conditions fail.
                else_node["current"] = instruction
                else_node["path"].append(instruction)
                if else_node["current"] == "R":
                    rejected_paths.append(else_node)
                elif else_node["current"] != "A":
                    queue.append(else_node)
                else:
                    accepted_paths.append(else_node)


def day19(example=False, reload=False):
    """Sort the machine parts."""
    if example:
        day = ("px{a<2006:qkq,m>2090:A,rfg}\n"
               "pv{a>1716:R,A}\n"
               "lnx{m>1548:A,A}\n"
               "rfg{s<537:gd,x>2440:R,A}\n"
               "qs{s>3448:A,lnx}\n"
               "qkq{x<1416:A,crn}\n"
               "crn{x>2662:A,R}\n"
               "in{s<1351:px,qqz}\n"
               "qqz{s>2770:qs,m<1801:hdj,R}\n"
               "gd{a>3333:R,R}\n"
               "hdj{m>838:A,pv}\n"
               "\n"
               "{x=787,m=2655,a=1222,s=2876}\n"
               "{x=1679,m=44,a=2067,s=496}\n"
               "{x=2036,m=264,a=79,s=2244}\n"
               "{x=2461,m=1339,a=466,s=291}\n"
               "{x=2127,m=1623,a=2188,s=1013}\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", cast=None, override=reload)
    load_workflow = True
    workflows = {}
    part_ratings = []
    # Puzzle parsing, put things into dictionaries.
    for line in puzzle:
        if not line:
            load_workflow = False
            continue
        if load_workflow:
            name, instruction = re.split("[{}]", line)[:-1]
            workflows[name] = instruction.split(",")
        else:
            ratings = line[1:-1].split(",")
            part_ratings.append({key: int(value) for key, value in (pair.split("=") for pair in ratings)})

    rejection_paths = []
    all_combinations = 4000**4
    rejected_combinations = rating_numbers = 0
    # Use a BFS to find and record all the rejection paths and the x,m,a,s ranges that would result in rejection.
    day19_bfs(workflows, rejection_paths)
    #  If the part matched any rejection path range it is not accepted.
    for part in part_ratings:
        if not any(all(part[key] in range(*rejection[key]) for key in "xmas") for rejection in rejection_paths):
            rating_numbers += sum(part.values())
    print(f"Part 1 the sum of rating numbers for all accepted parts is {rating_numbers}")
    #  For each rejection path calculate the number of combinations by multiplying each range together.
    for rejection in rejection_paths:
        if example:
            print(rejection)
        failed = [np.diff(x)[0] for x in (rejection[key] for key in "xmas")]
        rejected_combinations += np.prod(failed, dtype="uint64")
    accepted_combinations = all_combinations - rejected_combinations
    print(f"Part 2 there are {accepted_combinations:.0f} accepted combinations")
    for a in a_p:
        print(a)


class FlipFlop():  # %
    def __init__(self, children):
        self.parents = []
        self.children = children
        self.state = False
        self.pending_in = {}
        self.pending_out = {}
    def setup(self):
        for k in self.parents:
            self.pending_in[k] = []
        for k in self.children:
            self.pending_out[k] = []
    def update(self):
        for k in self.pending_in:
            for x in self.pending_in[k]:
                if x == False:
                    self.state = not self.state
                    for o in self.pending_out:
                        self.pending_out[o].append(self.state)
            self.pending_in[k]=[]


class Dummy():
    def __init__(self):
        self.parents = []
        self.children = None
        self.pending_in = {}
        self.pending_out = {}
        self.started=False
    def setup(self):
        for k in self.parents:
            self.pending_in[k] = []
    def update(self):
        for k in self.pending_in:
            for x in self.pending_in[k]:
                if x == False:
                    self.started=True
            self.pending_in[k] = []


class Broadcast():
    def __init__(self, children):
        self.parents = []
        self.children = children
        self.pending_in = {}
        self.pending_out = {}
    def button(self):
        for k in self.children:
            self.pending_out[k].append(False)
    def setup(self):
        for k in self.children:
            self.pending_out[k] = []


class Conjunction():  # &
    def __init__(self, children):
        self.parents = []
        self.children = children
        self.memory = {}
        self.pending_in = {}
        self.pending_out = {}
    def setup(self):
        for k in self.parents:
            self.memory[k] = False
            self.pending_in[k] = []
        for k in self.children:
            self.pending_out[k] = []
    def get_output(self):
        return not all(v == True for v in self.memory.values())
    def update(self):
        for k in self.pending_in:
            for x in self.pending_in[k]:
                self.memory[k] = x
                for o in self.pending_out:
                    self.pending_out[o].append(self.get_output())
            self.pending_in[k] = []


def status(d):
    for k,v in d.items():
        print(k)
        print(" in:",v.parents, v.pending_in)
        print(" out:",v.children, v.pending_out)
    print()


def process_in(modules):
    for name, module in modules.items():
        if hasattr(module, "update"):
            module.update()

def process_out(modules, counters):
    for name, module in modules.items():
        for k in module.pending_out:
            for x in module.pending_out[k]:
                modules[k].pending_in[name].append(x)
                counters[x] += 1
            module.pending_out[k]=[]

def run_it(modules, counters, debug=False):
    while True:
        process_out(modules, counters)
        if debug:
            print("process_out")
            status(modules)
            input()
        process_in(modules)
        if debug:
            print("process_in")
            status(modules)
            input()
        work = False
        for m in modules.values():
            for k,v in m.pending_out.items():
                if v:
                    work = True
                    break
        if not work:
            break
    return counters




def day20(example=False, reload=False):
    """1."""
    if example:
        day = """broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output
"""        
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    p = get_input(day, "\n", cast=None, override=reload)
    p1 = p2 = 0
    modules = {}
    # connect the modules.
    for l in p:
        name, connections = l.split(" -> ")
        connections = connections.split(", ")
        if name[1:] in modules:
            print(name)
            raise Exception()
        if name == "broadcaster":
            modules["broadcaster"] = Broadcast(connections)
        elif "%" in name:
            modules[name[1:]] = FlipFlop(connections)
        elif "&" in name:
            modules[name[1:]] = Conjunction(connections)
    # Wire it up
    for_testing = {}
    for k, v in modules.items():
        for c in v.children:
            if c not in modules:
                print(c, "not in modules?")
                for_testing[c] = Dummy()
                for_testing[c].parents.append(k)
            else:
                modules[c].parents.append(k)
    modules.update(for_testing)
    # set initial state:
    for v in modules.values():
        v.setup()
    #status(modules)
    counters = {True:0,False:0}
    dumb = {"dh":0,"mk":0,"vf":0,"rn":0}
    #for i in range(1000):
    while True:
        #print("Button",i)
        #input()
        but += 1
        counters[False] += 1
        modules["broadcaster"].button()
        #status(modules)
        run_it(modules, counters)
        print(but)
        if modules["rx"].started:
            break
    print(counters)
    print(counters[False] * counters[True])
