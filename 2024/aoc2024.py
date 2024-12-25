"""
Advent of Code.

Never did spend the time to work out how to get oAuth to work so this code expects you to
manually copy over your session cookie value.
Using a web browser inspect the cookies when logged into the Advent of Code website.
Copy the value from the "session" cookie into a text file called "session.txt"
"""

# import collections
import copy
# import curses  # pip install windows-curses
import functools
# import heapq
import inspect
import itertools
# import math
import pickle
import random
import re
import socket
import sortedcontainers
# import statistics
# import sys
# import time
from os import path

import cpmpy as cp
import networkx as nx
import numpy as np
# import pyglet
import requests  # type: ignore[import-untyped]

from collections import defaultdict

# Constants
_code_path = r"c:\AoC"
_offline = False
_year = 2024


def _check_internet(host="8.8.8.8", port=53, timeout=2):
    """
    Attempt to check for the firewall by connecting to Google's DNS.

    Args:
        host (str): A DNS server IP address, defaults to 8.8.8.8
        port (int): A port number, defaults to 53.
        timeout (int): A timeout value, defaults to 2.

    Returns:
        bool: True if DNS ping succeeds.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True  # noqa: TRY300
    except socket.error:  # noqa: UP024
        return False


def _pull_puzzle_input(day, seperator, cast=None):
    """
    Pull the puzzle data from the AOC website.

    Args:
        day (int, str): the AoC day puzzle input to fetch or an example puzzle string
        seperator (str, None): A string separator to pass into str.split when consuming the puzzle data.
            If None or "" don't try and split the puzzle input.
        cast (None, type): A Python function often a type cast (int, str, lambda) to be run against each data element.

    Returns:
        tuple of the puzzle data.

    Raises:
        Exception: if session file does not exist.
    """
    if _offline:
        with open(_code_path + r"\{}\day{}.txt".format(_year, day)) as file_handler:  # noqa: UP032, FURB101, PTH123
            data_list = file_handler.read().split(seperator)
    elif isinstance(day, str):  # An example string
        data_list = day.split(seperator)
    else:
        if not path.exists(_code_path + "/session.txt"):  # noqa: PTH110
            raise Exception("Using the web browser get the session cookie value\nand put it as a string in {}".format(_code_path + r"\session.txt"))  # noqa: W605, EM103, TRY002
        with open(_code_path + "/session.txt", 'r') as session_file:  # noqa: Q000, UP015, FURB101, PTH123
            session = session_file.read()
        # Check to see if behind the firewall.
        if _check_internet():  # noqa: SIM108
            proxy_dict = {}
        else:
            proxy_dict = {"http": "proxy-dmz.intel.com:911",
                          "https": "proxy-dmz.intel.com:912"}
        header = {"Cookie": "session={:s}".format(session.rstrip("\n"))}
        with requests.Session() as session:
            resp = session.get("https://adventofcode.com/{}/day/{}/input".format(_year, day), headers = header, proxies = proxy_dict)  # noqa: E251, UP032
            _ = resp.text.strip("\n")
            if resp.ok:
                if seperator in [None, ""]:  # noqa: PLR6201, SIM108
                    data_list = [resp.text]
                else:
                    data_list = resp.text.split(seperator)
            else:
                print("Warning website error")
                return ()

    if data_list[-1] == "":  # noqa: PLC1901
        data_list.pop(-1)
    if cast is not None:
        data_list = [cast(x) for x in data_list]
    return tuple(data_list)


# Cache the data in a pickle file.
def get_input(day, seperator, cast=None, override=False):
    """
    Fetch the daily puzzle information.

    If the puzzle data does not exist it attempts to pull it from the website.
    Caches the puzzle data into a pickle file so that re-runs don't have the performance
    penalty of fetching from the Advent Of Code website.

    Params:
        day (int, str): the AoC day puzzle input to fetch or a string of the puzzle example.
        seperator (str): A string separator to pass into str.split when consuming the puzzle data.
        cast (type, None): A Python function often a type cast (int, str, lambda) to be run against each data element.
        override (bool): True to re-download the puzzle input.

    Returns:
        tuple containing the puzzle data
    """
    if path.exists(_code_path + r"\{}\input.p".format(_year)):  # noqa: UP032, PTH110, SIM108
        puzzle_dict = pickle.load(open(_code_path + r"\{}\input.p".format(_year), "rb"))  # noqa: UP032, SIM115, PTH123, S301
    else:  # No pickle file, will need to make a new one.
        puzzle_dict = {}

    puzzle_input = puzzle_dict.get(day, None)

    if puzzle_input is None or override is True:
        puzzle_input = _pull_puzzle_input(day, seperator, cast)
        if isinstance(day, int):  # only save the full puzzle data to the pickle file.
            puzzle_dict[day] = puzzle_input
            pickle.dump(puzzle_dict, open(_code_path + r"\{}\input.p".format(_year), "wb"))  # noqa: UP032, SIM115, PTH123, S301
    return puzzle_input


def get_np_input(day, seperator, cast=None, splitter=None, dtype=None, override=False):
    """
    Wrap get_input and cast the allow casting the data type too.

    returns a numpy array instead of the tuple array that get_input does.

    Params:
        day (int, str): the AoC day puzzle input to fetch or a string of the puzzle example.
        seperator (str): A string separator to pass into str.split when consuming the puzzle data.
        cast (type, None): A Python function often a type cast (int, str, lambda) to be run against each data element.
        splitter (function, None): A splitter function to be called on the input data.
        dtype (str, None): The data type hint for numpy.
        override (bool): True to re-download the puzzle input.

    Returns:
        Numpy array

    """
    day_input = get_input(day, seperator, cast, override)
    if splitter is None:
        return np.array(day_input, dtype=dtype)
    temp = [splitter(x) for x in day_input]
    return np.array(temp, dtype=dtype)


def print_np(array):
    """Small script to print a numpy array to the console visually similar to the puzzles in AoC."""
    if array.dtype == np.dtype("<U1"):
        for row in array:
            print("".join(row))
    else:
        for row in array:
            print(np.array2string(row, separator="", max_line_width=600)[1:-1])


# Some code experiments in a visualization module instead of using curses.
# class Viz(pyglet.window.Window):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs# )
#
#     def on_draw(self):
#         self.clear()
#         label = pyglet.text.Label("Hello, world",font_name='Times New Roman',font_size=36,
#                                   x=self.width//2, y=self.height//2, anchor_x='center',anchor_y='center')
#         label.draw()
#
# def v():
#     #label = pyglet.text.Label("Hello, world",font_name='Times New Roman',font_size=36, anchor_x='center',anchor_y='center')
#     _ = Viz(512, 512, "Test",resizable=False)
#     pyglet.app.run()


class PointObject:
    """
    A point object to use with 2D arrays where y/row is the first index and x/column is the second.

    Useful when you want to turn the 2D map into a graph.
    """

    def __init__(self, y, x, shape=None, empty_shape=".", direction=None):
        """Set up the initial values."""
        self.x = x
        self.y = y
        self.contains = shape
        self.is_empty = self.contains == empty_shape
        self.direction = direction
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.up_right = None
        self.up_left = None
        self.down_right = None
        self.down_left = None

    def move(self, direction=None, steps=1):
        """Move the point object."""
        direction = self.direction if direction is None else direction
        if direction in {"u", "n", "up", "north"}:
            self.y -= steps
        if direction in {"d", "s", "down", "south"}:
            self.y += steps
        if direction in {"r", "e", "right", "east"}:
            self.x += steps
        if direction in {"l", "w", "left", "west"}:
            self.x -= steps
        if direction in {"ur", "ne", "up_right", "north_east"}:
            self.y -= steps
            self.x += steps
        if direction in {"ul", "nw", "up_left", "north_west"}:
            self.y -= steps
            self.x -= steps
        if direction in {"dr", "se", "down_right", "south_east"}:
            self.y += steps
            self.x += steps
        if direction in {"dl", "sw", "down_left", "south_west"}:
            self.y += steps
            self.x -= steps

    def position(self):
        """Return position tuple."""
        return (self.y, self.x)

    def p(self):
        """Return position tuple."""
        return (self.y, self.x)

    def show(self):
        """Display internal state."""
        print(f"({self.y},{self.x}) {self.contains}")
        if self.direction:
            print(f"direction: {self.direction}")
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

    def __abs__(self):
        """Return absolute value for each part."""
        return Coordinate(abs(x) for x in self)
    def __mul__(self, other):
        """Multiply a scaler with this coordinate."""
        return Coordinate(x * other for x in self)
    def __neg__(self):
        """Turn the coordinate negative."""
        return Coordinate(-1 * x for x in self)
    def __add__(self, other):
        """Add two coordinates or a coordinate and a tuple."""
        return Coordinate(x + y for x, y in zip(self, other, strict=True))
    def __sub__(self, other):
        """Subtract one coordinate from another."""
        return Coordinate(x - y for x, y in zip(self, other, strict=True))
    def __lt__(self, other):
        """Use to test if coordinate in 2D array."""
        return all(x < y for x, y in zip(self, other, strict=True))
    def __le__(self, other):
        """Use to test if coordinate in 2D array."""
        return all(x <= y for x, y in zip(self, other, strict=True))
    def __gt__(self, other):
        """Use to test if coordinate in 2D array."""
        return all(x > y for x, y, in zip(self, other, strict=True))
    def __ge__(self, other):
        """Use to test if coordinate in 2D array."""
        return all(x >= y for x, y, in zip(self, other, strict=True))
    def __setitem__(self, key, value):
        """Ok, look it really isn't a tuple."""
        self_list = list(self)
        self_list[key] = value
        return Coordinate(tuple(self_list))
    def manhattan_dist(self, other):
        """Calculate the manhattan distance between this coordinate and another."""
        return Coordinate(abs(x - y) for x, y in zip(self, other, strict=True))


# Dictionary to make walking the 2D maps easier.
move_dict = {"u": (-1, 0), "n": (-1, 0), "up": (-1, 0), "north": (-1, 0), "^": (-1, 0),
             "d": (1, 0), "s": (1, 0), "down": (1, 0), "south": (1, 0), "v": (1, 0),
             "r": (0, 1), "e": (0, 1), "right": (0, 1), "east": (0, 1), ">": (0, 1),
             "l": (0, -1), "w": (0, -1), "left": (0, -1), "west": (0, -1), "<": (0, -1),
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

_reverse = {"r": "l", "e": "w", "right": "left", "east": "west",
            "l": "r", "w": "e", "left": "right", "west": "east",
            "u": "d", "n": "s", "up": "down", "north": "south",
            "d": "u", "s": "n", "down": "up", "south": "north"}

turn_dict = {"r": _right, "right": _right, "cw": _right, "clockwise": _right,
             "l": _left, "left": _left, "ccw": _left, "counterclockwise": _left}


def dfs(graph, node):  # Example function for DFS
    """DFS search."""
    visited = set()
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor)


def bfs(graph, node):  # Example function for BFS
    """BFS search."""
    visited = set()
    queue = [node]

    while queue:          # Creating loop to visit each node
        this_node = queue.pop(0)
        print(this_node)

    for neighbor in graph[this_node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)


def day1_original(example=False):
    """So it begins."""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example == 1:
        day = ("3   4\n"
               "4   3\n"
               "2   5\n"
               "1   3\n"
               "3   9\n"
               "3   3\n")

    # print(day)
    p = get_input(day, "\n", None, override=False)
    l = []  # noqa: E741
    r = []
    for i in p:
        a, b = i.split("  ")
        l.append(int(a))
        r.append(int(b))
    # print(l)
    l.sort()
    r.sort()
    d = []
    for i in range(len(l)):
        d.append(abs(l[i] - r[i]))  # noqa: PERF401
    # print(d)
    print(sum(d))
    p2 = 0
    for i in range(len(l)):
        p2 += r.count(l[i]) * l[i]
    print(p2)


def day1(example=False):
    """So it begins."""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("3   4\n"
               "4   3\n"
               "2   5\n"
               "1   3\n"
               "3   9\n"
               "3   3\n")
    location = np.array(get_input(day, "\n", lambda x: (int(x.split()[0]), int(x.split()[1])), override=False))
    part_1 = np.sum(np.abs(location[:, 0][location[:, 0].argsort()] - location[:, 1][location[:, 1].argsort()]))
    print(f"Part 1: The total distance is {part_1}")
    part_2 = np.sum([x * np.count_nonzero(location[:, 1] == x) for x in np.nditer(location[:, 0])])
    print(f"Part 2: The similarity score is {part_2}")


def day2_part1_original(example=False):
    """Day 2."""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("7 6 4 2 1\n"
               "1 2 7 8 9\n"
               "9 7 6 2 1\n"
               "1 3 2 4 5\n"
               "8 6 4 4 1\n"
               "1 3 6 7 9")
    p1 = 0
    p = get_input(day, "\n", lambda x: list(map(int, x.split(" "))), override=False)
    for lvl in p:
        a = np.diff(lvl)
        a.sort()
        print()
        print(lvl)
        print(a)
        if a[-1] > 3:
            continue
        if a[0] < -3:
            continue
        if 0 in a:
            continue
        if np.where(a < 0)[0].size == a.size:
            print("safe neg")
            p1 += 1
        if np.where(a > 0)[0].size == a.size:
            print("safe pos")
            p1 += 1
    print(p1)


def safe_level_orig(lvl):
    """
    Check if the level is safe.

    Args:
        lvl (array): level array.

    Returns:
        bool
    """
    a = np.diff(lvl)
    a.sort()
    if a[-1] > 3:
        return False
    if a[0] < -3:
        return False
    if 0 in a:
        return False
    if np.where(a < 0)[0].size == a.size:
        return True
    if np.where(a > 0)[0].size == a.size:  # noqa: SIM103
        return True
    return False


def day2_part2_original(example=False):
    """Day 2."""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("7 6 4 2 1\n"
               "1 2 7 8 9\n"
               "9 7 6 2 1\n"
               "1 3 2 4 5\n"
               "8 6 4 4 1\n"
               "1 3 6 7 9")
    p2 = 0
    p = get_input(day, "\n", lambda x: list(map(int, x.split(" "))), override=True)
    for lvl in p:
        if safe_level_orig(lvl):
            p2 += 1
        else:
            for i in range(len(lvl)):
                temp = copy.deepcopy(lvl)
                temp.pop(i)
                if safe_level_orig(temp):
                    p2 += 1
                    break
    print(p2)


def safe_level(np_array):
    """
    Check if the levels on a report are 'safe'.

    Args:
        np_array (np.array): The report to check

    Returns:
        bool: True if safe.
    """
    safe = False
    dampener_safe = False
    pause = False
    # print(np.diff(np_array))

    pos_lvl_diff = np.isin(np.diff(np_array), [1, 2, 3])
    neg_lvl_diff = np.isin(np.diff(np_array), [-1, -2, -3])
    if np.all(pos_lvl_diff) or np.all(neg_lvl_diff):
        safe = dampener_safe = True
    if np.count_nonzero(np.invert(pos_lvl_diff)) == 1:
        unsafe_location = np.where(np.invert(pos_lvl_diff))[0]
        print("unsafe at:", unsafe_location)
        temp = np.diff(np.delete(np_array, unsafe_location + 1))
        print(temp)
        if np.all(np.isin(temp, [-1, -2, -3])) or np.all(np.isin(temp, [1, 2, 3])):
            dampener_safe = True
        else:
            pause = True
    if np.count_nonzero(np.invert(neg_lvl_diff)) == 1:
        unsafe_location = np.where(np.invert(neg_lvl_diff))[0]
        print("unsafe at:", unsafe_location)
        temp = np.diff(np.delete(np_array, unsafe_location + 1))
        print(temp)
        if np.all(np.isin(temp, [-1, -2, -3])) or np.all(np.isin(temp, [1, 2, 3])):
            dampener_safe = True
        else:
            pause = True
    print(safe, dampener_safe)
    if pause:
        _ = input()
    return safe, dampener_safe


def day2(example=False, override=False):
    """Day 2."""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("7 6 4 2 1\n"
               "1 2 7 8 9\n"
               "9 7 6 2 1\n"
               "1 3 2 4 5\n"
               "8 6 4 4 1\n"
               "1 3 6 7 9")
    p1 = p2 = 0
    puzzle = get_input(day, "\n", lambda x: np.array(list(map(int, x.split(" ")))), override=override)
    for level in puzzle:
        for check in [[1, 2, 3], [-1, -2, -3]]:
            differences = np.diff(level)
            bad_locations = np.where(np.isin(differences, check, invert=True))[0]
            if bad_locations.size == 0:
                p1 += 1
                p2 += 1
            if (bad_locations.size == 1 and
                (np.where(np.isin(np.diff(np.delete(level, bad_locations[0])), check, invert=True))[0].size == 0 or
                np.where(np.isin(np.diff(np.delete(level, bad_locations[0] + 1)), check, invert=True))[0].size == 0)):
                p2 += 1
            if (bad_locations.size == 2 and np.diff(bad_locations)[0] == 1 and
                differences[bad_locations[0]] + differences[bad_locations[1]] in check):
                p2 += 1

    print(f"Part 1: The number of safe reports is {p1}")
    print(f"Part 2: The number of safe reports using the 'Problem Dampener' is {p2}")


def mul(a, b):
    """
    Multiply for eval use.

    Returns:
        a * b
    """
    return a * b


def day3_eval(example=True, override=False):
    """
    Day 3.

    Args:
        example (bool): Use the example input.
        override (boot): Override the stored get_input data.
    """
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = """xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))"""
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    r = re.compile(r"mul\([0-9]*,[0-9]*\)|do\(\)|don't\(\)")
    do = True
    for line in p:
        f = r.findall(line)
        print(f)
        for i in f:
            if i == "do()":
                do = True
                continue
            if i == "don't()":
                do = False
                continue
            p1 += eval(i)  # noqa: S307
            if do:
                p2 += eval(i)  # noqa: S307
    print(p1)
    print(p2)


def day3(example=True, override=False):
    """
    Day 3.

    Args:
        example (bool): Use the example input.
        override (boot): Override the stored get_input data.
    """
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = """xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))"""
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    r = re.compile(r"mul\([0-9]*,[0-9]*\)|do\(\)|don't\(\)")
    do = True
    for line in p:
        f = r.findall(line)
        print(f)
        for i in f:
            if i == "do()":
                do = True
                continue
            if i == "don't()":
                do = False
                continue
            a = int(i.split("(")[1].split(",")[0])
            b = int(i.split(",")[1].split(")")[0])
            print(i, a, b)
            p1 += (a * b)
            if do:
                p2 += (a * b)
    print(p1)
    print(p2)


def day4(example=False, override=False):
    """Day 4."""
    day: int | str
    day = ("MMMSXXMASM\n"
           "MSAMXMSMSA\n"
           "AMXSXMAAMM\n"
           "MSAMASMSMX\n"
           "XMASAMXAMM\n"
           "XXAMMXXAMA\n"
           "SMSMSASXSS\n"
           "SAXAMASAAA\n"
           "MAMMMXMMMM\n"
           "MXMXAXMASX\n")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p1 = p2 = 0
    puzzle = get_input(day, "\n", None, override=override)
    the_map = []
    for line in puzzle:
        the_map.append(list(line))  # noqa: PERF401
    the_xs: list[PointObject] = []
    # Part 1
    # Find the X's
    for y, line in enumerate(puzzle):
        for x, c in enumerate(line):
            if c == "X":
                the_xs.append(PointObject(y, x))
    # Check for XMAS
    for p in the_xs:
        for md in ["u", "d", "l", "r", "ur", "ul", "dl", "dr"]:
            t = PointObject(*p.p())
            t.move(md)
            if t.y not in range(len(the_map)) or t.x not in range(len(the_map[0])) or the_map[t.y][t.x] != "M":
                continue
            t.move(md)
            if t.y not in range(len(the_map)) or t.x not in range(len(the_map[0])) or the_map[t.y][t.x] != "A":
                continue
            t.move(md)
            if t.y not in range(len(the_map)) or t.x not in range(len(the_map[0])) or the_map[t.y][t.x] != "S":
                continue
            p1 += 1
    # Part 2
    # Find the A's
    the_as = []
    for y, line in enumerate(puzzle):
        for x, c in enumerate(line):
            if c == "A":
                the_as.append(PointObject(y, x))
    # Check for MAS
    for a in the_as:
        ur = PointObject(*a.p())
        ur.move("ur")
        ul = PointObject(*a.p())
        ul.move("ul")
        dr = PointObject(*a.p())
        dr.move("dr")
        dl = PointObject(*a.p())
        dl.move("dl")
        if ur.y in range(len(the_map)) and ur.x in range(len(the_map)) and \
           ul.y in range(len(the_map)) and ul.x in range(len(the_map)) and \
           dl.y in range(len(the_map)) and dl.x in range(len(the_map)) and \
           dl.y in range(len(the_map)) and dl.x in range(len(the_map)):
            ul_l = the_map[ul.y][ul.x]
            ur_l = the_map[ur.y][ur.x]
            dl_l = the_map[dl.y][dl.x]
            dr_l = the_map[dr.y][dr.x]
            if ((ul_l == "M" and dr_l == "S") or (ul_l == "S" and dr_l == "M")) and \
               ((ur_l == "M" and dl_l == "S") or (ur_l == "S" and dl_l == "M")):
                p2 += 1
    print(p1)
    print(p2)


def day5(example=False, override=False):
    """Day 5."""
    day: int | str
    day = ("47|53\n97|13\n97|61\n97|47\n75|29\n61|13\n75|53\n29|13\n97|29\n"
           "53|29\n\n61|53\n\n97|53\n61|29\n47|13\n75|47\n97|75\n47|61\n75|61\n"
           "47|29\n75|13\n53|13"
           "\n"
           "75,47,61,53,29\n"
           "97,61,53,29,13\n"
           "75,29,13\n"
           "75,97,47,61,53\n"
           "61,13,29\n"
           "97,13,75,29,47\n")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    r = []
    u = []
    for line in p:
        if "|" in line:
            a, b = line.split("|")
            r.append((int(a), int(b)))
        if "," in line:
            u.append(list(map(int, line.split(","))))
    for x in u:
        invalid = False
        while True:
            changed = False
            for a, b in r:
                if a in x and b in x:
                    i_a = x.index(a)
                    i_b = x.index(b)
                    if i_a >= i_b:
                        invalid = True
                        changed = True
                        x[i_a] = b
                        x[i_b] = a
            if not changed:
                break

        if invalid:
            p2 += x[len(x) // 2]
        else:
            p1 += x[len(x) // 2]
    print(p1)
    print(p2)


rules: set[tuple[int, ...]] = set()


def _page_sort(a, b):
    global rules  # noqa: PLW0602
    val = 0
    if (a, b) in rules:
        val = -1
    elif (b, a) in rules:
        val = 1
    return val


def day5_sort(example=False, override=False):
    """Day 5."""
    day: int | str
    day = ("47|53\n97|13\n97|61\n97|47\n75|29\n61|13\n75|53\n29|13\n97|29\n"
           "53|29\n\n61|53\n\n97|53\n61|29\n47|13\n75|47\n97|75\n47|61\n75|61\n"
           "47|29\n75|13\n53|13"
           "\n"
           "75,47,61,53,29\n"
           "97,61,53,29,13\n"
           "75,29,13\n"
           "75,97,47,61,53\n"
           "61,13,29\n"
           "97,13,75,29,47\n")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    global rules  # noqa: PLW0603
    rules = set()
    u = []
    for line in p:
        if "|" in line:
            rules.add(tuple(map(int, line.split("|"))))
        if "," in line:
            u.append(np.array(line.split(","), dtype=int))
    for x in u:
        n = np.array(sorted(x, key=functools.cmp_to_key(_page_sort)))
        if (n == x).all():
            p1 += x[x.size // 2]
        else:
            p2 += n[n.size // 2]
    print(p1)
    print(p2)


def day5_bryce(override=False):
    """Bryce's solution to day 5 for time comparison."""
    from collections import defaultdict  # noqa: PLC0415
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    rules = defaultdict(list)
    updates = []
    for item in p:
        if "|" in item:
            page, later_page = item.split("|")
            page = int(page)
            later_page = int(later_page)
            rules[page].append(later_page)
        if "," in item:
            updates.append(list(map(int, item.split(","))))

    total_sum = 0
    for update in updates:
        still_good = True
        for i, page in enumerate(update):
            for later_page in update[i + 1:]:
                if later_page not in rules[page]:
                    still_good = False
                    break
            if not still_good:
                break
        if still_good:
            total_sum += update[(len(update) - 1) // 2]
    print("P1:", total_sum)

    total_sum = 0
    for update in updates:
        sorting_list = list(update)
        changed = False
        for page in update:
            index = sorting_list.index(page)
            for later_page in sorting_list[index + 1:]:
                if later_page not in rules[page]:
                    sorting_list.remove(later_page)
                    sorting_list.insert(index, later_page)
                    index += 1
                    changed = True
        if changed:
            total_sum += sorting_list[(len(sorting_list) - 1) // 2]
    print("P2:", total_sum)


def day6(example=False, override=False):
    """Day 6."""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("....#.....\n"
               ".........#\n"
               "..........\n"
               "..#.......\n"
               ".......#..\n"
               "..........\n"
               ".#..^.....\n"
               "........#.\n"
               "#.........\n"
               "......#...\n")
    a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    a = np.pad(a, 1, mode="constant", constant_values="E")
    loc = np.where(a == "^")
    g = PointObject(loc[0][0], loc[1][0], direction="u")
    pos = set()
    while a[g.p()] != "E":
        if a[g.p()] == "#":
            g.move(steps=-1)
            g.direction = _right[g.direction]
        pos.add(g.p())
        g.move()
    print(f"Part 1: {len(pos)}")
    # Brute force by putting an obstruction in each spot along the guard path.
    p2 = 0
    for y, x in pos:
        if a[y][x] == "#" or a[y][x] == "^":
            continue
        g = PointObject(loc[0][0], loc[1][0], direction="u")
        a[y][x] = "#"
        this_route = set()
        # print(f"Checking: ({x},{y})")
        while a[g.p()] != "E":
            if a[g.p()] == "#":
                g.move(steps=-1)
                g.direction = _right[g.direction]
            now = (g.y, g.x, g.direction)
            if now in this_route:
                p2 += 1
                break
            this_route.add(now)
            g.move()
        a[y][x] = "."
    print(f"Part 2: {p2}")


def day7(example=False, override=False):
    """Day 7."""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("190: 10 19\n3267: 81 40 27\n83: 17 5\n156: 15 6\n7290: 6 8 6 15\n"
               "161011: 16 10 13\n192: 17 8 14\n21037: 9 7 18 13\n292: 11 6 16 20\n")
    import operator  # noqa: PLC0415
    puzzle = get_input(day, "\n", None, override=override)
    equations = []
    logic = [operator.add, operator.mul, lambda x, y: int(str(x) + str(y))]
    for line in puzzle:
        a, b = line.split(":")
        equations.append((int(a), list(map(int, b.split()))))
    for part, j in [("Part 1: ", 2), ("Part 2:", 3)]:
        answer = 0
        for test_value, numbers in equations:
            for operations in itertools.product(logic[:j], repeat=len(numbers) - 1):
                i = 1
                this_val = numbers[0]
                for operation in operations:
                    this_val = operation(this_val, numbers[i])
                    i += 1
                    if this_val > test_value:  # Early exit operations only make things 'bigger'
                        break
                if this_val == test_value:
                    # print(" Matched: ", this_val)
                    answer += this_val
                    break  # Only add once.
        print(part, answer)


def day8(example=False, override=False):
    """Day 8."""
    day: int | str = ("............\n........0...\n.....0......\n.......0....\n"
                      "....0.......\n......A.....\n............\n............\n"
                      "........A...\n.........A..\n............\n............")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    city = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    frequencies = set([city[x] for x in zip(*np.where(city != "."), strict=True)])  # noqa: C403
    p1_antinodes: set[Coordinate] = set()
    p2_antinodes: set[Coordinate] = set()
    for frequency in frequencies:
        antennas = [Coordinate(x) for x in zip(*np.where(city == frequency), strict=True)]
        for one, two in itertools.combinations(antennas, 2):
            p2_antinodes.add(one)
            p2_antinodes.add(two)
            delta = one - two
            p1_flag = True
            new = one
            while True:
                new += delta
                if new[0] in range(city[0].size) and new[1] in range(city[:, 0].size):
                    p2_antinodes.add(new)
                    if p1_flag:
                        p1_antinodes.add(new)
                        p1_flag = False
                else:
                    break
            p1_flag = True
            new = two
            while True:
                new -= delta
                if new[0] in range(city[0].size) and new[1] in range(city[:, 0].size):
                    p2_antinodes.add(new)
                    if p1_flag:
                        p1_antinodes.add(new)
                        p1_flag = False
                else:
                    break
    print("Part 1: ", len(p1_antinodes))
    print("Part 2: ", len(p2_antinodes))
    # for p in p2_antinodes:
    #     city[p]="#"
    # print_np(city)


def day9(example=False, override=False):
    """Day 9."""
    day: int | str = """2333133121414131402"""
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    day = 9
    p = get_input(day, "\n", None, override=override)
    p = list(p[0])
    # print(p)
    # print(len(p))
    # a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    files = []
    free = []
    val = 0
    while p:
        size = int(p.pop(0))
        if size:
            files.append((size, val))
            val += 1
        try:
            free = p.pop(0)
        except Exception:  # noqa: BLE001
            break
        if free:
            files.append((int(free), "F"))
    idx = 0
    check = 0
    # _ = input()
    while files:
        # print("before")
        # print(files[:3])
        # print(files[-3:])
        # print("check: ",check, "idx: ", idx)
        t = files.pop(0)
        # print("t:",t)
        if t[1] == "F" and t[0] == 0:
            continue
        if t[1] == "F":  # free space
            e = None
            while True:
                if not files:
                    break
                e = files.pop()
                # print("e: ",e)
                if e[1] != "F":
                    break
            if e:
                if e[1] == "F":
                    print("Error about to insert blank")
                    _ = input()
                if t[0] == e[0]:
                    files = [e, files]
                elif t[0] > e[0]:
                    files = [(t[0] - e[0], "F"), files]
                    files = [e, files]
                if t[0] < e[0]:
                    files = [(t[0], e[1]), files]
                    files.append((e[0] - t[0], e[1]))
        else:
            for _ in range(t[0]):
                check += t[1] * idx
                idx += 1
        # print("after")
        # print(files[:3])
        # print(files[-3:])
        # print("check: ",check, "idx: ", idx)
        print(len(files))
        # _ = input()
    print(check)
    # print(files)


def day9_part2(example=False, override=False):
    """Day 9."""
    day: int | str
    day = """2333133121414131402"""
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    p = list(p[0])
    # print(p)
    # print(len(p))
    # a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    files = []
    free = []
    val = 0
    while p:
        size = int(p.pop(0))
        if size:
            files.append((size, val))
            val += 1
        try:
            free = p.pop(0)
        except Exception:  # noqa: BLE001
            break
        if free:
            files.append((int(free), "F"))
    idx = 0
    check = 0
    # print(files)
    for idx in reversed(range(val)):
        print(idx)
        for i in reversed(range(len(files))):
            if files[i][1] == idx:
                for j in range(i):
                    if files[j][1] == "F" and files[j][0] >= files[i][0]:
                        x = files.pop(i)
                        e = files.pop(j)
                        files.insert(j, x)
                        files.insert(i, (x[0], "F"))
                        if e[0] > x[0]:
                            r = e[0] - x[0]
                            files.insert(j + 1, (r, "F"))
                        break
        # print(files)
        # _ = input()
    # print()
    # print(files)
    pos = 0
    for f in files:
        for _ in range(f[0]):
            if f[1] != "F":
                check += f[1] * pos
            pos += 1
    print(check)


def trail(graph, node):
    """DFS trail search."""
    visited = set()
    found = set()
    if node not in visited:
        visited.add(node)
        for neighbor in ["u", "d", "l", "r"]:
            t = PointObject(*node.p())
            t.move(neighbor)
            if t.y in range(graph[0].size) and t.x in range(graph[:, 0].size) and graph[t.p()] - 1 == graph[node.p()]:
                found = found.union(trail(graph, t))
                if graph[t.p()] == 9:
                    found.add(t.p())
    return found


def day10(example=False, override=False):
    """Day 10."""
    day: int | str
    day = ("89010123\n"
           "78121874\n"
           "87430965\n"
           "96549874\n"
           "45678903\n"
           "32019012\n"
           "01329801\n"
           "10456732\n")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    a = get_np_input(day, "\n", splitter=list, dtype=int, override=override)
    p1 = 0
    start = [PointObject(y, x) for y, x in zip(*np.where(a == 0), strict=True)]
    for s in start:
        p1 += len(trail(a, s))
    print(p1)


def trail_bfs(graph, node):  # Example function for BFS
    """BFS search."""
    visited = set(node.p())
    queue = [node]
    paths = 0
    while queue:          # Creating loop to visit each node
        this_node = queue.pop(0)
        for neighbor in "news":
            t = PointObject(*this_node.p())
            t.move(neighbor)
            if t.y in range(graph[0].size) and t.x in range(graph[:, 0].size) and graph[t.p()] - 1 == graph[this_node.p()]:
                if graph[t.p()] == 9:
                    paths += 1
                else:
                    queue.append(t)
                    visited.add(t.p())
    return paths


def day10_2(example=False, override=False):
    """Day 10."""
    day: int | str
    day = ("89010123\n"
           "78121874\n"
           "87430965\n"
           "96549874\n"
           "45678903\n"
           "32019012\n"
           "01329801\n"
           "10456732\n")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    # Code to make the three path example work.
    b = np.full((a[0].size, a[:, 0].size), -1, dtype=int)
    for i in range(10):
        b[np.where(a == str(i))] = i
    p2 = 0
    start = [PointObject(y, x) for y, x in zip(*np.where(b == 0), strict=True)]
    for s in start:
        p2 += trail_bfs(b, s)
    print(p2)


def day11_part1_orig(example=False, override=False):
    """Day 11."""
    day: int | str
    day = """125 17"""
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, " ", int, override=override)
    print(p)
    for i in range(75):
        t = []
        print(i)
        for s in p:
            if s == 0:
                t.append(1)
            elif len(str(s)) % 2 == 0:
                st = str(s)
                t.extend([int(st[:len(st) // 2]), int(st[len(st) // 2:])])
            else:
                t.append(s * 2024)
        p = copy.deepcopy(t)
    print(len(p))


def day11(example=False, override=False):
    """Day 11."""
    day: int | str
    day = """125 17"""
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, " ", int, override=override)
    d = defaultdict(lambda: 0)
    for x in p:
        d[x] += 1
    for i in range(75):
        t = defaultdict(lambda: 0)
        for x, v in d.items():
            if x == 0:
                t[1] += v
            elif len(str(x)) % 2 == 0:
                n = str(x)
                h = len(n) // 2
                t[int(n[:h])] += v
                t[int(n[h:])] += v
            else:
                t[x * 2024] += v
        d = copy.deepcopy(t)
        if i == 24:
            print("Part 1: ", sum(d.values()))
    print("Part 2: ", sum(d.values()))


#def plots_bfs(graph, node):  # Example function for BFS
#    """BFS search."""
#    visited = set([node])
#    queue = [node]
#    paths = 0
#    corner = 0
#    while queue:          # Creating loop to visit each node
#        this_node = queue.pop(0)
#
#        # ul corner
#        if graph[*this_node+move_dict["u"]] != graph[*this_node] and graph[*this_node+move_dict["l"]] != graph[*this_node]:
#            #print("ul ex:",this_node)
#            corner += 1
#        # ur corner
#        if graph[*this_node+move_dict["u"]] != graph[*this_node] and graph[*this_node+move_dict["r"]] != graph[*this_node]:
#            corner += 1
#            #print("ur ex:",this_node)
#        # dr corner
#        if graph[*this_node+move_dict["d"]] != graph[*this_node] and graph[*this_node+move_dict["r"]] != graph[*this_node]:
#            corner += 1
#            #print("dr ex:",this_node)
#        # dl corner
#        if graph[*this_node+move_dict["d"]] != graph[*this_node] and graph[*this_node+move_dict["l"]] != graph[*this_node]:
#            corner += 1
#            #print("dl ex:",this_node)
#        # Interior corners
#        # ul
#        if (graph[*this_node+move_dict["d"]] == graph[*this_node] and
#            graph[*this_node+move_dict["r"]] == graph[*this_node] and
#            graph[*this_node+move_dict["d"]+move_dict["r"]] != graph[*this_node]):
#            corner += 1
#            #print("ul in:",this_node)
#        # ur
#        if (graph[*this_node+move_dict["d"]] == graph[*this_node] and
#            graph[*this_node+move_dict["l"]] == graph[*this_node] and
#            graph[*this_node+move_dict["d"]+move_dict["l"]] != graph[*this_node]):
#            corner += 1
#            #print("ur in:",this_node)
#        # dl
#        if (graph[*this_node+move_dict["u"]] == graph[*this_node] and
#            graph[*this_node+move_dict["r"]] == graph[*this_node] and
#            graph[*this_node+move_dict["u"]+move_dict["r"]] != graph[*this_node]):
#            corner += 1
#            #print("dl in:",this_node)
#        # dr
#        if (graph[*this_node+move_dict["u"]] == graph[*this_node] and
#            graph[*this_node+move_dict["l"]] == graph[*this_node] and
#            graph[*this_node+move_dict["u"]+move_dict["l"]] != graph[*this_node]):
#            corner += 1
#            #print("dr in:",this_node)
#
#        for neighbor in ["u","d","l","r"]:
#            t=this_node + move_dict[neighbor]
#            if t not in visited and t[0] in range(graph[0].size) and t[1] in range(graph[:,0].size) and graph[*t] == graph[*this_node]:
#                queue.append(t)
#                visited.add(t)
#
#    return visited, corner


# def day12(example=False, override=False):
#    """
#    Day 12.
#
#    Returns:
#        list
#    """
#    day: int | str
#    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
#    if example == 1:
#        day = """AAAA\nBBCD\nBBCC\nEEEC"""
#    if example == 2:
#        day = """OOOOO\nOXOXO\nOOOOO\nOXOXO\nOOOOO"""
#    if example == 3:
#        day = ("RRRRIICCFF\n"
#               "RRRRIICCCF\n"
#               "VVRRRCCFFF\n"
#               "VVRCCCJFFF\n"
#               "VVVVCJJCFE\n"
#               "VVIVCCJJEE\n"
#               "VVIIICJJEE\n"
#               "MIIIIIJJEE\n"
#               "MIIISIJEEE\n"
#               "MMMISSJEEE")
#    a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
#    plots = a[0].size * a[:,0].size
#    a = np.pad(a, 1, mode="constant", constant_values=".")
#    b=copy.deepcopy(a)
#    #print_np(a)
#    plot_types = set([a[x] for x in zip(*np.where(a!="."), strict=True)])
#    #print(plot_types)
#    visited = set()
#    sizes = []
#    p2 = 0
#    return a
#    while len(visited) < plots:
#        for t in plot_types:
#            l = set([Coordinate(x) for x in zip(*np.where(a==t), strict=True)])
#            while l:
#                temp = l.pop()
#                v,c = plots_bfs(a, temp)
#                #print(a[temp],"area: ",len(v),"corners: ",c)
#                sizes.append(v)
#                l = l.difference(v)
#                visited = visited.union(v)
#                p2 += len(v) * c
#                #print(temp)
#                #print(v)
#                #print(visited)
#                for x in v:
#                    a[x]=0
#                #print_np(a)
#                #_=input()
#        #print("v",len(visited))
#        #print(plots)
#        #_=input()
#    #print(sizes)
#    #p1=0
#    #for s in sizes:
#    #    if len(s) == 1:
#    #        p1+=4
#    #    elif len(s) == 2:
#    #        p1+=2*4
#    #    else:
#    #        t=s.pop()
#    #        print(a[t])
#    #        s.add(t)
#    #        #print(s)
#    ##        p=Polygon(s)
#    #        sides = len(p.exterior.coords)-1
#    #        print("sides: ",sides)
#    #        print("area: ", len(s))
#    #        p1+=len(s)*sides
#
#    print(p2)
#    #print_np(b)
#    #return sizes

def plots_bfs(graph, node):
    """BFS search."""
    visited = {node}
    queue = [node]
    corners = 0
    perimeter = 0
    while queue:  # Creating loop to visit each node
        this_node = queue.pop(0)
        this_char = graph[this_node]
        this_3x3 = graph[this_node[0] - 1:this_node[0] + 2, this_node[1] - 1:this_node[1] + 2]
        # Perimeter (edge) check
        e_mask = np.array([[1, 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]])
        e_masked = np.ma.masked_array(this_3x3, e_mask)
        perimeter += 4 - np.count_nonzero(e_masked == this_char)
        # Corner checks.
        mask = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [1, 1, 1]])
        diag_mask = np.array([[0, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
        for _ in range(4):
            masked = np.ma.masked_array(this_3x3, mask)
            diag_masked = np.ma.masked_array(this_3x3, diag_mask)
            # Exterior corner
            if np.count_nonzero(masked == this_char) == 0:
                corners += 1
            # Interior corner
            if np.count_nonzero(masked == this_char) == 2 and \
               np.count_nonzero(diag_masked == this_char) == 0:
                corners += 1
            this_3x3 = np.rot90(this_3x3)

        for neighbor in ["u", "d", "l", "r"]:
            t = this_node + move_dict[neighbor]
            if t not in visited and t[0] in range(graph[0].size) and \
               t[1] in range(graph[:, 0].size) and graph[t] == graph[this_node]:
                queue.append(t)
                visited.add(t)

    return visited, corners, perimeter


def day12(example=False, override=False):
    """Day 12."""
    day: int | str
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example == 1:
        day = """AAAA\nBBCD\nBBCC\nEEEC"""
    if example == 2:
        day = """OOOOO\nOXOXO\nOOOOO\nOXOXO\nOOOOO"""
    if example == 3:
        day = ("RRRRIICCFF\n"
               "RRRRIICCCF\n"
               "VVRRRCCFFF\n"
               "VVRCCCJFFF\n"
               "VVVVCJJCFE\n"
               "VVIVCCJJEE\n"
               "VVIIICJJEE\n"
               "MIIIIIJJEE\n"
               "MIIISIJEEE\n"
               "MMMISSJEEE")
    if example == 4:
        day = ("----\n"
               "-OOO\n"
               "-O-O\n"
               "OO-O\n"
               "-OOO")
    a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    total_plots = a[0].size * a[:, 0].size
    a = np.pad(a, 1, mode="constant", constant_values=".")
    plot_types = {a[x] for x in zip(*np.where(a != "."), strict=True)}
    total_visited = set()
    p1 = p2 = 0
    while len(total_visited) < total_plots:
        for t in plot_types:
            plots = {Coordinate(x) for x in zip(*np.where(a == t), strict=True)}.difference(total_visited)
            while plots:
                temp = plots.pop()
                visited, corners, perimeter = plots_bfs(a, temp)
                # print(a[*temp], "area: ", len(visited), "sides: ", corners, "perimeter: ", perimeter)
                p1 += len(visited) * perimeter
                p2 += len(visited) * corners
                total_visited = total_visited.union(visited)
                plots = {Coordinate(x) for x in zip(*np.where(a == t), strict=True)}.difference(total_visited)
    print("Part 1: ", p1)
    print("Part 2: ", p2)


def day13(example=False, override=False):
    """Day 13."""
    day: int | str
    day = ("Button A: X+94, Y+34\n"
           "Button B: X+22, Y+67\n"
           "Prize: X=8400, Y=5400\n"
           "\n"
           "Button A: X+26, Y+66\n"
           "Button B: X+67, Y+21\n"
           "Prize: X=12748, Y=12176\n"
           "\n"
           "Button A: X+17, Y+86\n"
           "Button B: X+84, Y+37\n"
           "Prize: X=7870, Y=6450\n"
           "\n"
           "Button A: X+69, Y+23\n"
           "Button B: X+27, Y+71\n"
           "Prize: X=18641, Y=10279")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    puzzle = get_input(day, "\n", None, override=override)
    for p, offset in enumerate([0, 10000000000000]):
        ans = 0
        for i in range(0, len(puzzle), 4):
            ax = int(puzzle[i].split("X")[1].split(",")[0])
            ay = int(puzzle[i].split("Y")[1].strip())
            bx = int(puzzle[i + 1].split("X")[1].split(",")[0])
            by = int(puzzle[i + 1].split("Y")[1].strip())
            tx = int(puzzle[i + 2].split("X=")[1].split(",")[0]) + offset
            ty = int(puzzle[i + 2].split("Y=")[1].strip()) + offset
            a, b = cp.intvar(lb=0, ub=10000000000000, shape=2)
            m = cp.Model((a * ax) + (b * bx) == tx, (a * ay) + (b * by) == ty)
            m.minimize((3 * a) + b)
            if m.solve():
                # print("A: ", a.value(), "B: ", b.value())
                ans += 3 * a.value() + b.value()
        print(f"Part {p + 1}: ", ans)


def day14(example=False, override=False):
    """Day 14."""
    day: int | str
    day = ("p=0,4\nv=3,-3\np=6,3\nv=-1,-3\np=10,3\nv=-1,2\np=2,0\nv=2,-1\np=0,0\nv=1,3\n"
           "p=3,0\nv=-2,-2\np=7,6\nv=-1,-3\np=3,0\nv=-1,-2\np=9,3\nv=2,3\np=7,3\nv=-1,2\n"
           "p=2,4 v=2,-3\np=9,5\nv=-3,-3")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    r = {}
    i = 0
    for i, l in enumerate(p):  # noqa: E741
        px = int(l.split("p=")[1].split(",")[0])
        py = int(l.split("p=")[1].split(",")[1].split()[0])
        vx = int(l.split("v=")[1].split(",")[0])
        vy = int(l.split("v=")[1].split(",")[1].strip())
        r[i] = [Coordinate((py, px)), Coordinate((vy, vx))]
    for x, y in r.items():
        print(x, y)
    if example:
        w = 11
        t = 7
    else:
        w = 101
        t = 103
    for s in range(10000000):
        for a, v in r.items():
            # print(a,v)
            r[a][0] = v[0] + v[1]
        a = np.zeros([t, w], dtype=int)
        for v in r.values():
            y = v[0][0] % t
            x = v[0][1] % w
            # print(a[0].size, a[:,0].size)
            # print(v,y,x)
            a[y, x] += 1
        # print_np(a)
        # a=np.delete(a, w//2, 1)
        # a=np.delete(a, t//2, 0)
        # print()
        # print_np(a)
        # b=[M for subA in np.split(a,2,axis=0) for M in np.split(subA,2,axis=1)]
        # for x in b:
        #    p1.append(np.sum(x))
        for x in a:
            if np.sum(x) > 30:
                c = np.full([t, w], fill_value=" ", dtype=str)
                c[np.where(a != 0)] = "#"
                print_np(c)
                print("@" * 100, s)
                _ = input()
                break
        for x in a.T:
            if np.sum(x) > 30:
                c = np.full([t, w], fill_value=" ", dtype=str)
                c[np.where(a != 0)] = "#"
                print_np(c)
                print("@" * 100, s)
                _ = input()
                break
    # print(np.prod(p1))


def push(sub_array):
    """Push the box."""
    free = np.where(sub_array == ".")[0][0]
    wall = np.where(sub_array == "#")[0][0]
    if free < wall:  # Can push
        sub_array[:free + 1] = np.roll(sub_array[:free + 1], shift=1)


def day15_part1(example=False):
    """Day 15."""
    day: int | str
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example == 1:
        day = ("########\n"
               "#..O.O.#\n"
               "##@.O..#\n"
               "#...O..#\n"
               "#.#.O..#\n"
               "#...O..#\n"
               "#......#\n"
               "########\n"
               "\n"
               "<^^>>>vv<v>>v<<")
    if example == 2:
        day = ("#######\n"
               "#...#.#\n"
               "#.....#\n"
               "#..OO@#\n"
               "#..O..#\n"
               "#.....#\n"
               "#######\n"
               "\n"
               "<vv<<^^<<^^")
    if example == 3:
        day = ("##########\n"
               "#..O..O.O#\n"
               "#......O.#\n"
               "#.OO..O.O#\n"
               "#..O@..O.#\n"
               "#O#..O...#\n"
               "#O..O..O.#\n"
               "#.OO.O.OO#\n"
               "#....O...#\n"
               "##########\n"
               "\n"
               "<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^\n"
               "vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v\n"
               "><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<\n"
               "<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^\n"
               "^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><\n"
               "^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^\n"
               ">^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^\n"
               "<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>\n"
               "^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>\n"
               "v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^")
    puzzle = get_input(day, "\n", None, override=True)
    movement_index = puzzle.index("")
    warehouse = np.array([list(x) for x in puzzle[:movement_index]], dtype=str)
    movements = puzzle[movement_index:]
    for moves in movements:
        for move in moves:
            # prev = np.copy(warehouse)
            ry, rx = np.argwhere(warehouse == "@")[0]
            if move == "^" and "." in warehouse[:, rx][:ry + 1]:
                push(np.flip(warehouse[:, rx][:ry + 1]))
            elif move == "v" and "." in warehouse[:, rx][ry:]:
                push(warehouse[:, rx][ry:])
            elif move == "<" and "." in warehouse[ry][:rx + 1]:
                push(np.flip(warehouse[ry][:rx + 1]))
            elif move == ">" and "." in warehouse[ry][rx:]:
                push(warehouse[ry][rx:])
    # print_np(warehouse)
    y, x = np.where(warehouse == "O")
    print("Part 1: ", np.sum(y) * 100 + np.sum(x))


def move_graph(node, direction, warehouse, graph):
    """Create the move graph."""
    next_node = (node[0] + move_dict[direction], warehouse[node[0] + move_dict[direction]])
    graph.add_edge(node, next_node)
    if next_node[1] in "[]":
        lr = "l" if next_node[1] == "]" else "r"
        other_node = (next_node[0] + move_dict[lr], warehouse[next_node[0] + move_dict[lr]])
        move_graph(other_node, direction, warehouse, graph)
        move_graph(next_node, direction, warehouse, graph)


def wide_push(graph, warehouse):
    """Move wide boxes."""
    if all(node == "." for node in [v[1] for v, d in graph.out_degree() if d == 0]):
        while graph.size():
            leaves = [v for v, d in graph.out_degree() if d == 0]
            for node in leaves:
                parents = graph.predecessors(node)
                if parents:
                    for p in parents:
                        warehouse[node[0]] = p[1]
                        warehouse[p[0]] = "."
                else:
                    warehouse[node[0]] = "."
                graph.remove_node(node)


def day15_part2(example=False):
    """Day 15."""
    day: int | str
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example == 1:
        day = ("########\n"
               "#..O.O.#\n"
               "##@.O..#\n"
               "#...O..#\n"
               "#.#.O..#\n"
               "#...O..#\n"
               "#......#\n"
               "########\n"
               "\n"
               "<^^>>>vv<v>>v<<")
    if example == 2:
        day = ("#######\n"
               "#...#.#\n"
               "#.....#\n"
               "#..OO@#\n"
               "#..O..#\n"
               "#.....#\n"
               "#######\n"
               "\n"
               "<vv<<^^<<^^")
    if example == 3:
        day = ("##########\n"
               "#..O..O.O#\n"
               "#......O.#\n"
               "#.OO..O.O#\n"
               "#..O@..O.#\n"
               "#O#..O...#\n"
               "#O..O..O.#\n"
               "#.OO.O.OO#\n"
               "#....O...#\n"
               "##########\n"
               "\n"
               "<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^\n"
               "vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v\n"
               "><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<\n"
               "<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^\n"
               "^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><\n"
               "^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^\n"
               ">^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^\n"
               "<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>\n"
               "^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>\n"
               "v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^")
    puzzle = get_input(day, "\n", None, override=True)
    movement_index = puzzle.index("")
    warehouse = []
    for i in range(movement_index):
        temp = []
        for c in puzzle[i]:
            if c == ".":
                temp += [".", "."]
            elif c == "O":
                temp += ["[", "]"]
            elif c == "#":
                temp += ["#", "#"]
            elif c == "@":
                temp += ["@", "."]
            else:
                print(f"Unknown character {c}")
                return
        warehouse.append(temp)
    warehouse = np.array(warehouse, dtype=str)
    movements = puzzle[movement_index:]
    for moves in movements:
        for move in moves:
            # prev = np.copy(warehouse)
            ry, rx = np.argwhere(warehouse == "@")[0]
            pos = Coordinate((ry, rx))
            graph = nx.DiGraph()
            graph.add_edge(((0, 0), "."), (pos, "@"))
            if move == "^" and "." in warehouse[:, rx][:ry + 1]:
                graph.add_edge(((0, 0), "."), (pos, "@"))
                move_graph((pos, "@"), move, warehouse, graph)
                # prev_g = copy.deepcopy(graph)
                wide_push(graph, warehouse)
            elif move == "v" and "." in warehouse[:, rx][ry:]:
                move_graph((pos, "@"), move, warehouse, graph)
                # prev_g = copy.deepcopy(graph)
                wide_push(graph, warehouse)
            elif move == "<" and "." in warehouse[ry][:rx + 1]:
                push(np.flip(warehouse[ry][:rx + 1]))
            elif move == ">" and "." in warehouse[ry][rx:]:
                push(warehouse[ry][rx:])

            # error = False
            # for p in np.argwhere(warehouse == "["):
            #   if warehouse[p[0],p[1]+1] != "]":
            #     error = True
            # for p in np.argwhere(warehouse == "]"):
            #   if warehouse[p[0],p[1]-1] != "[":
            #     error = True
            # if error:
            #     print(move)
            #     print_np(prev)
            #     print()
            #     print_np(warehouse)
            #     nx.draw(prev_g, with_labels=True)
            #     import matplotlib.pyplot as plt
            #     plt.show()
            #     _=input()
    # print_np(warehouse)
    y, x = np.where(warehouse == "[")
    print("Part 2: ", np.sum(y) * 100 + np.sum(x))


def day16(example=False, override=False):
    """Day 16."""
    day: int | str
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example == 1:
        day = ("###############\n"
               "#.......#....E#\n"
               "#.#.###.#.###.#\n"
               "#.....#.#...#.#\n"
               "#.###.#####.#.#\n"
               "#.#.#.......#.#\n"
               "#.#.#####.###.#\n"
               "#...........#.#\n"
               "###.#.#####.#.#\n"
               "#...#.....#.#.#\n"
               "#.#.#.###.#.#.#\n"
               "#.....#...#.#.#\n"
               "#.###.#.#.#.#.#\n"
               "#S..#.....#...#\n"
               "###############")
    if example == 2:
        day = ("#################\n"
               "#...#...#...#..E#\n"
               "#.#.#.#.#.#.#.#.#\n"
               "#.#.#.#...#...#.#\n"
               "#.#.#.#.###.#.#.#\n"
               "#...#.#.#.....#.#\n"
               "#.#.#.#.#.#####.#\n"
               "#.#...#.#.#.....#\n"
               "#.#.#####.#.###.#\n"
               "#.#.#.......#...#\n"
               "#.#.###.#####.###\n"
               "#.#.#...#.....#.#\n"
               "#.#.#.#####.###.#\n"
               "#.#.#.........#.#\n"
               "#.#.#.#########.#\n"
               "#S#.............#\n"
               "#################")
    maze = get_np_input(day, seperator="\n", cast=None, splitter=list, dtype=str, override=override)
    visited = set()
    start = Coordinate(np.argwhere(maze == "S")[0])
    end = Coordinate(np.argwhere(maze == "E")[0])
    turn = {"h": "v", "v": "h"}
    moves = {"h": "we", "v": "ns"}
    graph = nx.Graph()
    # BFS to map the maze into a graph.
    queue = [(start, "h")]
    while queue:
        cur_node = queue.pop(0)
        visited.add(cur_node)
        graph.add_node(cur_node)
        rotated = (cur_node[0], turn[cur_node[1]])
        graph.add_edge(cur_node, rotated, weight=1000)  # Turning costs 1000
        if rotated not in visited:
            queue.append(rotated)
        for d in moves[cur_node[1]]:
            next_node = (cur_node[0] + move_dict[d], cur_node[1])
            if not ((0, 0) <= next_node[0] < maze.shape) or \
               next_node in visited or \
               maze[next_node[0]] == "#":
                continue
            graph.add_edge(cur_node, next_node, weight=1)
            queue.append(next_node)
    # Figure out ending in which direction is the shortest.
    v = nx.shortest_path_length(graph, source=(start, "h"), target=(end, "v"), weight="weight")
    h = nx.shortest_path_length(graph, source=(start, "h"), target=(end, "h"), weight="weight")
    print(f"Part 1: {min(v, h)}")
    # Count all positions along the shortest paths.
    end_dir = "v" if v < h else "h"
    paths = nx.all_shortest_paths(graph, source=(start, "h"), target=(end, end_dir), weight="weight")
    seats = set()
    for this_path in paths:
        seats = seats.union({x[0] for x in this_path})
    print(f"Part 2: {len(seats)}")


class Comp:
    """Day 17 computer class."""
    def __init__(self, a=0, b=0, c=0, instructions=""):
        """Set up the computer class."""
        self.a = a
        self.b = b
        self.c = c
        self.program = []
        self.instructions = instructions
        self.output = []
        self.ip = None
        self.match = False

    def reinit(self, a):
        """Reset the computer."""
        self.a = a
        self.b = 0
        self.c = 0
        self.ip = 0
        self.output = []
        self.match = False

    def opcode(self, x, y):
        """Process the opcode."""
        match x:
            case 0:  # adv
                # print(f"adv: {self.a // 2**y} = {self.a} // 2**{y}")
                self.a //= 2**y
            case 1:  # bxl
                # print(f"bxl: {self.b ^ y} = {self.b} ^ {y}")
                self.b ^= y
            case 2:  # bst
                # print(f"bst: {y % 8} = {y} % 8")
                self.b = y % 8
            case 3:  # jnz
                if self.a:
                    # print("jump from ", self.ip, " to ", y)
                    self.ip = y - 2
            case 4:  # bxc
                # print(f"bxc: {self.c ^ self.b} = {self.c} ^ {self.b}")
                self.b = self.c ^ self.b
            case 5:  # out
                # print(f"out: {y % 8} = {y} % 8")
                self.output.append(y % 8)
            case 6:  # bdv
                # print(f"bdv: {self.a // 2**y} = {self.a} // 2**{y}")
                self.b = self.a // 2**y
            case 7:  # cdv
                # print(f"cdv: {self.a // 2**y} = {self.a} // 2**{y}")
                self.c = self.a // 2**y
        self.ip += 2

    def decode(self):
        """Decode the instruction string."""
        commands = self.instructions.split(",")
        self.program = list(map(int, commands))
        self.ip = 0

    def run(self):
        """Run the program."""
        # self.decode()
        while 0 <= self.ip < len(self.program) - 1:
            # print(f"before a: {self.a} b: {self.b} c: {self.c}")
            x, y = self.program[self.ip : self.ip + 2]
            # print(self.ip, x, y)
            if x in {0, 2, 5, 6, 7}:
                match y:
                    case 4:
                        y = self.a
                    case 5:
                        y = self.b
                    case 6:
                        y = self.c
                    case 7:
                        error = "illegal combo operand."
                        raise Exception(error)  # noqa: TRY002
            self.opcode(x, y)
            if self.output != self.program[:len(self.output)]:
                # print(f"mismatch {self.output} {self.program}")
                break
            # print(f"after a: {self.a} b: {self.b} c: {self.c}")
            # _ = input()
        if self.output == self.program:
            self.match = True
        # print(self.output)


@functools.lru_cache(maxsize=2**21)
def _comp_segment(a, b, c, opcodes, operands):
    """
    Process the segment.

    Returns:
        list: output
    """
    output = []
    ip = 0
    while 0 <= ip < len(opcodes):
        x = opcodes[ip]
        y = operands[ip]
        if x in {0, 2, 5, 6, 7}:
            match y:
                case 4:
                    y = a
                case 5:
                    y = b
                case 6:
                    y = c
        match x:
            case 0:  # adv
                a //= 2**y
            case 1:  # bxl
                b ^= y
            case 2:  # bst
                b = y % 8
            case 3:  # jnz
                error = "ERROR! Jump detected"
                raise Exception(error)  # noqa: TRY002
            case 4:  # bxc
                b = c ^ b
            case 5:  # out
                output.append(y % 8)
            case 6:  # bdv
                b = a // 2**y
            case 7:  # cdv
                c = a // 2**y
        ip += 1
    return a, b, c, output


def _day17_thread2(p, r):
    print(p, r)
    opcodes = [p[x] for x in range(0, len(p), 2)]
    operands = [p[x] for x in range(1, len(p), 2)]
    # print(opcodes)
    for i in r:
        a = i
        b = 0
        c = 0
        ip = 0
        output = []
        while 0 <= ip < len(opcodes):
            # print("next",ip, opcodes[ip])
            if opcodes[ip] == 3:
                if a:
                    ip = operands[ip]
                else:
                    ip += 1
            elif 3 in opcodes[ip:]:
                jmp_loc = opcodes[ip:].index(3)
                a, b, c, more_output = _comp_segment(a, b, c, tuple(opcodes[ip:jmp_loc + ip]),
                                                     tuple(operands[ip:jmp_loc + ip]))
                output.extend(more_output)
                ip += jmp_loc
                # print(ip, output)
                # _ = input()
            else:
                a, b, c, more_output = _comp_segment(a, b, c, tuple(opcodes[ip:]), tuple(operands[ip:]))
                output.extend(more_output)
                ip += len(opcodes[ip:])
            if output != p[:len(output)]:
                break
        if output == p:
            print(i)
        # print(output)


def _day17_thread(p, r):
    print(p, r)
    for i in r:
        a = i
        b = 0
        c = 0
        output = []
        program = p
        ip = 0
        while 0 <= ip < len(program):
            x, y = program[ip : ip + 2]
            if x in {0, 2, 5, 6, 7}:
                match y:
                    case 4:
                        y = a
                    case 5:
                        y = b
                    case 6:
                        y = c
            match x:
                case 0:  # adv
                    a //= 2**y
                case 1:  # bxl
                    b ^= y
                case 2:  # bst
                    b = y % 8
                case 3:  # jnz
                    if a:
                        ip = y - 2
                case 4:  # bxc
                    b = c ^ b
                case 5:  # out
                    output.append(y % 8)
                case 6:  # bdv
                    b = a // 2**y
                case 7:  # cdv
                    c = a // 2**y
            ip += 2
            if output != program[:len(output)]:
                break
        if output == program:
            print("yes", i)


def day17(example=False, override=False, start=0, nmax=8):
    """Day 17."""
    day: int | str
    day = ("Register A: 2024\n"
           "Register B: 0\n"
           "Register C: 0\n"
           "\n"
           "Program: 0,3,5,4,3,0")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", lambda x: x.split(":")[-1], override=override)
    c = Comp(*map(int, p[:3]), p[4])
    c.decode()
    from concurrent import futures  # noqa: PLC0415
    threads = 16
    with futures.ProcessPoolExecutor(max_workers=threads) as executor:
        running = []
        for thread in range(threads):
            # r = range(2024, 2025)
            r = range(start + thread, 10**nmax + thread + start, threads)
            f = executor.submit(_day17_thread2, list(map(int, p[4].split(","))), r)
            running.append(f)
        futures.wait(running, return_when=futures.ALL_COMPLETED)
    # while not c.match:
    #     p2 += 1
    #     if p2 % 1000:
    #         print(p2)
    #     c.reinit(p2)
    #     c.run()
    #     # _ = input()
    # return c
    # print(p2)
    # print(",".join(map(str, c.output)))
    # Made it to 24500000000 wihtout finding the answer to 17p2.


def day18(example=False, override=False):
    """Day 18."""
    day: int | str
    day = ("5,4\n4,2\n4,5\n3,0\n2,1\n6,3\n2,4\n1,5\n0,6\n3,3\n2,6\n5,1\n1,2\n5,5\n2,5\n"
           "6,5\n1,4\n0,4\n6,4\n1,1\n6,1\n1,0\n0,5\n1,6\n2,0")
    grid_size = 7
    part_1_bytes = 12
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
        grid_size = 71
        part_1_bytes = 1024
    p = get_input(day, "\n", lambda x: Coordinate(tuple(map(int, x.split(",")))), override=override)
    graph = nx.Graph()
    for x, y in itertools.product(range(grid_size), range(grid_size)):
        graph.add_node(Coordinate((x, y)))
    for node in graph.nodes:
        for direction in "news":
            neighbor = node + move_dict[direction]
            if neighbor[0] in range(grid_size) and neighbor[1] in range(grid_size):
                graph.add_edge(neighbor, node)
    for i, x in enumerate(p):
        graph.remove_node(x)
        if i == part_1_bytes:
            print("Part 1:", nx.shortest_path_length(graph, source=(0, 0), target=(grid_size - 1, grid_size - 1)))
        if not nx.has_path(graph, source=(0, 0), target=(grid_size - 1, grid_size - 1)):
            print(f"Part 2: {x[0]},{x[1]}")
            break


def day19(example=False, override=False):
    """Day 19."""
    day: int | str
    day = ("r, wr, b, g, bwu, rb, gb, br\n"
           "\n"
           "brwrr\n"
           "bggr\n"
           "gbbr\n"
           "rrbgbr\n"
           "ubwu\n"
           "bwurrg\n"
           "brgr\n"
           "bbrgwb")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    p1 = p2 = 0
    towels = p[0].split(", ")
    towels.sort(key=len, reverse=True)
    # print(towels)
    print(len(p[2:]))
    for orig in p[2:]:
        l = copy.deepcopy(orig)
        towels.sort(key=len, reverse=True)
        for t in towels:
            l = l.replace(t,"!")
        if all(c == "!" for c in l):
            p1 += 1
        else:
            for i in range(1000):
                l = copy.deepcopy(orig)
                random.shuffle(towels)
                for t in towels:
                    l = l.replace(t, "!")
                if all(c == "!" for c in l):
                    print(orig, l)
                    p1 += 1
                    break
    print(p1)
    print(p2)


def day19_2(example=False, override=False):
    """Day 19."""
    day: int | str
    day = ("r, wr, b, g, bwu, rb, gb, br\n"
           "\n"
           "brwrr\n"
           "bggr\n"
           "gbbr\n"
           "rrbgbr\n"
           "ubwu\n"
           "bwurrg\n"
           "brgr\n"
           "bbrgwb")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    p2 = 0
    towels = p[0].split(", ")
    towels.sort(key=len, reverse=True)
    towels = set(towels)
    # print(towels)
    most_stripes = len(towels[0])
    loop = 200
    print(len(p[2:]))
    for arrangement in p[2:]:
        visited = set()
        queue = [p[2]]
        while queue:
            this_towel = queue.pop(0)
            print(this_node)
    for neighbor in graph[this_node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)


def day20(example=False, override=False):
    """Day 20."""
    day: int | str
    day = ("###############\n"
           "#...#...#.....#\n"
           "#.#.#.#.#.###.#\n"
           "#S#...#.#.#...#\n"
           "#######.#.#.###\n"
           "#######.#.#...#\n"
           "#######.#.###.#\n"
           "###..E#...#...#\n"
           "###.#######.###\n"
           "#...###...#...#\n"
           "#.#####.#.###.#\n"
           "#.#...#.#.#...#\n"
           "#.#.#.#.#.#.###\n"
           "#...#...#...###\n"
           "###############")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    # p = get_input(day, "\n", None, override=override)
    a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    print_np(a)
    p1 = p2 = 0
    g = nx.Graph()
    print(range(a[0,:].size))
    print(range(a[:,0].size))
    for y, x in itertools.product(range(a[0,:].size), range(a[:, 0].size)):
        g.add_node(Coordinate((y, x)))
    for node in g.nodes:
        for direction in "news":
            neighbor = node + move_dict[direction]
            if neighbor[0] in range(a[0,:].size) and neighbor[1] in range(a[:, 0].size):
                g.add_edge(neighbor, node)
    print(g)
    source = Coordinate(np.argwhere(a=="S")[0])
    target = Coordinate(np.argwhere(a=="E")[0])
    for n in np.argwhere(a=="#"):
        g.remove_node(Coordinate(n))
    print(g)
    orig = nx.shortest_path_length(g, source=source, target=target)
    d = {}
    for y, x in itertools.product(range(1,a[0,:].size-1), range(1,a[:,0].size-1)):
        node = Coordinate((y,x))
        if a[node] == "#":
            g.add_node(node)
            for direction in "news":
                neighbor = node + move_dict[direction]
                g.add_edge(neighbor, node)
            this_cheat = nx.shortest_path_length(g, source=source, target=target)
            saved = orig - this_cheat
            if saved > 0:
                if saved in d:
                    d[saved] += 1
                else:
                    d[saved] = 1
            g.remove_node(node)
        #_=input()
    print(d)
    for k,v in d.items():
        if k >= 100:
            p1 += v
    print(p1)


def _day20_thread(g,a,checked,spaces,source,target,orig,to_save):
    p1 = 0
    #print(spaces)
    found = set()
    for space in spaces:
        window = 20
        min_y = max(0, space[0]-window)
        max_y = min(space[0]+window+1, a[0,:].size)
        min_x = max(0, space[1]-window)
        max_x = min(space[1]+window+1, a[:,0].size)
        #print(min_y, max_y, min_x, max_x)
        in_range = [Coordinate((y+min_y,x+min_x)) for y,x in np.argwhere(a[min_y:max_y,min_x:max_x] == ".")]
        #print_np(a[min_y:max_y,min_x:max_x])
        #print(space, len(in_range), len(checked))
        #_=input()
        for two in in_range:
            if (space, two) in checked or (two, space) in checked:
                #print("in")
                continue
            checked.add((space,two))
            man = sum(space.manhattan_dist(two))
            if 1 < man <= 20:
                g.add_edge(space, two)
                this_cheat = nx.shortest_path_length(g, source=source, target=target) + man - 1
                if orig - this_cheat >= to_save:
                    found.add((space,two))
                    found.add((two,space))
                g.remove_edge(space, two)
    return found


def day20_2(example=False, to_save=50, override=False):
    """Day 20."""
    day: int | str
    day = ("###############\n"
           "#...#...#.....#\n"
           "#.#.#.#.#.###.#\n"
           "#S#...#.#.#...#\n"
           "#######.#.#.###\n"
           "#######.#.#...#\n"
           "#######.#.###.#\n"
           "###..E#...#...#\n"
           "###.#######.###\n"
           "#...###...#...#\n"
           "#.#####.#.###.#\n"
           "#.#...#.#.#...#\n"
           "#.#.#.#.#.#.###\n"
           "#...#...#...###\n"
           "###############")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    # p = get_input(day, "\n", None, override=override)
    a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    #print_np(a)
    p1 = p2 = 0
    g = nx.Graph()
    #print(range(a[0,:].size))
    #print(range(a[:,0].size))
    for y, x in itertools.product(range(a[0,:].size), range(a[:, 0].size)):
        g.add_node(Coordinate((y, x)))
    for node in g.nodes:
        for direction in "news":
            neighbor = node + move_dict[direction]
            if neighbor[0] in range(a[0,:].size) and neighbor[1] in range(a[:, 0].size):
                g.add_edge(neighbor, node)
    source = Coordinate(np.argwhere(a=="S")[0])
    target = Coordinate(np.argwhere(a=="E")[0])
    a[source]="."
    a[target]="."
    for n in np.argwhere(a=="#"):
        g.remove_node(Coordinate(n))
    orig = nx.shortest_path_length(g, source=source, target=target)
    print(orig,g)
    d = {}
    spaces = [Coordinate(p) for p in np.argwhere(a==".")]
    #print(spaces)
    checked = set()
    results = []
    found = set()
    from concurrent import futures  # noqa: PLC0415
    threads = 18
    with futures.ProcessPoolExecutor(max_workers=threads) as executor:
        running = []
        for thread in range(threads):
            s = [spaces[x] for x in range(threads - thread, len(spaces)-thread, threads)]
            t = copy.deepcopy(g)
            f = executor.submit(_day20_thread, copy.deepcopy(g), a, checked, s, source, target, orig, to_save)
            running.append(f)
        futures.wait(running, return_when=futures.ALL_COMPLETED)
        for _ in futures.as_completed(running):
            found = found.union(_.result())
    print(len(found)//2)


def day20_rework(example=False, override=False):
    """Day 20."""
    day: int | str
    day = ("###############\n"
           "#...#...#.....#\n"
           "#.#.#.#.#.###.#\n"
           "#S#...#.#.#...#\n"
           "#######.#.#.###\n"
           "#######.#.#...#\n"
           "#######.#.###.#\n"
           "###..E#...#...#\n"
           "###.#######.###\n"
           "#...###...#...#\n"
           "#.#####.#.###.#\n"
           "#.#...#.#.#...#\n"
           "#.#.#.#.#.#.###\n"
           "#...#...#...###\n"
           "###############")
    at_least = 50
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
        at_least = 100
    # p = get_input(day, "\n", None, override=override)
    a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    print(f"Numpy array is ({a[0, :].size},{a[:, 0].size})")
    p1 = p2 = 0
    g = nx.Graph()
    for y, x in itertools.product(range(a[:, 0].size), range(a[0, :].size)):
        g.add_node(Coordinate((y, x)))
    cheats = set()
    for node in g.nodes:
        for distance in range(1,3):
            for direction in "news":
                neighbor = node + (move_dict[direction] * distance)
                if (0, 0) <= neighbor < (a[:, 0].size, a[0, :].size):
                    g.add_edge(neighbor, node, weight=distance)
                    #if distance > 1:
                    #    cheats.add(

    source = Coordinate(np.argwhere(a == "S")[0])
    target = Coordinate(np.argwhere(a == "E")[0])
    a[source] = "."
    a[target] = "."
    for n in np.argwhere(a == "#"):
        g.remove_node(Coordinate(n))
    print(g)
    spaces = map(Coordinate, np.argwhere(a == "."))
    space_pairs = [(one, two) for one, two in itertools.combinations(spaces, 2) if 1 < sum(one.manhattan_dist(two)) <= 20]
    print(f"Pairs to check is {len(list(space_pairs))}")
    return None
    orig_time = nx.shortest_path_length(g, source=source, target=target)
    d = defaultdict(int)
    left = len(space_pairs)
    for one, two in space_pairs:
        if left % 100 == 0:
            print(left)
        g.add_edge(one, two)
        saved_time = orig_time - (nx.shortest_path_length(g, source=source, target=target) + sum(one.manhattan_dist(two)) - 1)
        if saved_time > 0:
            d[saved_time] += 1
        g.remove_edge(one, two)
        left -= 1
    for k in sorted(d.keys()):
        print(f"{d[k]} cheats that save {k}")
        if k >= at_least:
            p1 += d[k]
    print(p1)


def numeric_decode(move):
    seq = []
    if move[1] < 0 and move[0] < 0:  # Move up before left
        seq.extend(["^"]*abs(move[0]))
        seq.extend(["<"]*abs(move[1]))
    elif move[0] > 0 and move[1] > 0:  # Move right before down
        seq.extend([">"]*abs(move[1]))
        seq.extend(["v"]*abs(move[0]))
    else:
        seq.extend(decode(move))
    seq.extend(["A"])
    return seq

def directional_decode(move):
    seq = []
    if move[1] < 0 and move[0] > 0:  # Move down before left
        seq.extend(["v"]*abs(move[0]))
        seq.extend(["<"]*abs(move[1]))
    elif move[0] < 0 and move[1] > 0:  # Move right before up
        seq.extend([">"]*abs(move[1]))
        seq.extend(["^"]*abs(move[0]))
    else:
        seq.extend(decode(move))
    seq.extend(["A"])
    return seq

def decode(move):
    seq = []
    v = "^" if move[0] < 0 else "v"
    h = "<" if move[1] < 0 else ">"
    seq.extend([v]*abs(move[0]))
    seq.extend([h]*abs(move[1]))
    return seq


def day21(example=False, override=False):
    """Day 21."""
    day: int | str
    day = """379A
"""    
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    keypad = {"7": Coordinate((0,0)), "8": Coordinate((0,1)), "9":   Coordinate((0,2)),
              "4": Coordinate((1,0)), "5": Coordinate((1,1)), "6":   Coordinate((1,2)),
              "1": Coordinate((2,0)), "2": Coordinate((2,1)), "3":   Coordinate((2,2)),
                                      "0": Coordinate((3,1)), "A": Coordinate((3,2))}
    directional = {             "^": Coordinate((0,1)), "A": Coordinate((0,2)),  # noqa: E201
        "<": Coordinate((1,0)), "v": Coordinate((1,1)), ">": Coordinate((1,2))}
    indirection = 2
    p1= 0
    for code in p:
        sequences = []
        cur_pos = Coordinate((3,2))
        seq = []
        for c in code:
            move = keypad[c] - cur_pos
            cur_pos = keypad[c]
            seq.extend(directional_decode(move))
            print(c," ","".join(seq))
        print("".join(seq))
        sequences.append(seq)
        for _ in range(indirection):
            cur_pos = Coordinate((0,2))
            this_seq = []
            for c in sequences[-1]:
                this_seq.extend(directional_decode(directional[c] - cur_pos))
                cur_pos = directional[c]
            print("".join(this_seq))
            sequences.append(this_seq)
        print("".join(sequences[-1]))
        print(code, " ", len(sequences[-1]), "*", int(code.strip("A")))
        p1 += len(sequences[-1])*int(code.strip("A"))
    print(p1)


@functools.cache
def _next_secret(num):
    num ^= (num * 64)
    num %= 16777216
    num ^= (num // 32)
    num %= 16777216
    num ^= (num * 2048)
    num %= 16777216
    return num


def day22(example=False, override=False):
    """Day 22."""
    day: int | str
    day = "1\n2\n3\n2024"
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    puzzle = get_input(day, "\n", int, override=override)
    p1 = 0
    bananas = defaultdict(int)
    for initial_secret in puzzle:
        prev = initial_secret % 10
        n = _next_secret(initial_secret)
        changes = tuple()
        first_seq = set()
        for _ in range(1, 4):
            changes += tuple([(n % 10) - prev])
            prev = n % 10
            n = _next_secret(n)
        for _ in range(4, 2000):
            this_banana = n % 10
            changes += tuple([this_banana - prev])
            prev = this_banana
            if changes not in first_seq:
                bananas[changes] += this_banana
                first_seq.add(changes)
            changes = changes[1:]
            n = _next_secret(n)
        p1 += n
    print(f"Part 1: {p1}")
    print(f"Part 2: {max(bananas.values())}")


def day23(example=False, override=False):
    """Day 23."""
    day: int | str
    day = ("kh-tc\nqp-kh\nde-cg\nka-co\nyn-aq\nqp-ub\ncg-tb\nvc-aq\ntb-ka\nwh-tc\nyn-cg\nkh-ub\n"
           "ta-co\nde-co\ntc-td\ntb-wq\nwh-td\nta-ka\ntd-qp\naq-cg\nwq-ub\nub-vc\nde-ta\nwq-aq\n"
           "wq-vc\nwh-yn\nka-de\nkh-ta\nco-tc\nwh-qp\ntb-vc\ntd-yn")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    puzzle = get_input(day, "\n", None, override=override)
    g = nx.Graph()
    things = set()
    for connection in puzzle:
        g.add_edge(*connection.split("-"))
    cliques = [clique for clique in nx.find_cliques(g) if len(clique) >= 3]
    for clique in cliques:
        for triangle in itertools.combinations(clique, 3):
            if any(computer[0] == "t" for computer in triangle):
                things.add(tuple(sorted(triangle)))
    print(f"Part 1: {len(things)}")
    print("Part 2:", ",".join(sorted(sorted(cliques, key=len)[-1])))


def day24(example=False, override=False):
    """Day 24."""
    day: int | str
    day = """x00: 0
x01: 1
x02: 0
x03: 1
x04: 0
x05: 1
y00: 0
y01: 0
y02: 1
y03: 1
y04: 0
y05: 1

x00 AND y00 -> z05
x01 AND y01 -> z02
x02 AND y02 -> z01
x03 AND y03 -> z03
x04 AND y04 -> z04
x05 AND y05 -> z00
"""    
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    p1 = p2 = 0
    orig_d = dict()
    zs = []
    x = 0
    y = 0
    c_loc = p.index("") + 1
    for i in p[:c_loc-1]:
        name, value = i.split(": ")
        orig_d[name] = int(value)
        v = int(value) << int(name[1:])
        if i[0] == "x":
            x |= v
        elif i[0] == "y":
            y |= v
    print(x, y, x+y)
    final = x+y
    for i in p[c_loc:]:
        things = i.split()
        #print(things)
        for x in range(0,5,2):
            if things[x][0] == "z":
                zs.append(things[x])
    #print(zs)
    d = copy.deepcopy(orig_d)
    counter = 0
    while any(d.get(z) is None for z in zs):
        for i in p[c_loc:]:
            left, op, right, _, dest = i.split()
            #print(left, d.get(left), op, right, d.get(right), dest)
            if d.get(dest):
                continue
            if d.get(left, None) is not None and d.get(right, None) is not None:
                if op == "OR":
                    d[dest] = d[left] | d[right]
                elif op == "AND":
                    d[dest] = d[left] & d[right]
                elif op == "XOR":
                    d[dest] = d[left] ^ d[right]
        counter += 1
        #print(d)
        #_=input()
    ans = [str(d[z]) for z in sorted(zs, reverse=True)]
    this_ans = int("".join(ans), 2)
    print("".join(ans))
    print(this_ans)
    print(counter)


def day24_2(example=False, override=False):
    """Day 24."""
    day: int | str
    day = """x00: 0
x01: 1
x02: 0
x03: 1
x04: 0
x05: 1
y00: 0
y01: 0
y02: 1
y03: 1
y04: 0
y05: 1

x00 AND y00 -> z05
x01 AND y01 -> z02
x02 AND y02 -> z01
x03 AND y03 -> z03
x04 AND y04 -> z04
x05 AND y05 -> z00
"""    
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    p1 = p2 = 0
    orig_d = dict()
    zs = []
    x = 0
    y = 0
    c_loc = p.index("") + 1
    for i in p[:c_loc-1]:
        name, value = i.split(": ")
        orig_d[name] = int(value)
        v = int(value) << int(name[1:])
        if i[0] == "x":
            x |= v
        elif i[0] == "y":
            y |= v
    print(x, y, x+y)
    final = x+y
    for i in p[c_loc:]:
        things = i.split()
        #print(things)
        for x in range(0,5,2):
            if things[x][0] == "z":
                zs.append(things[x])
    #print(zs)
    dests = []
    for i in p[c_loc:]:
        dests.append(i.split()[-1])
    for eight in itertools.combinations(dests, 8):
        for one in itertools.combinations(eight, 2):
            eight = set(eight).difference(one)
            for two in itertools.combinations(eight, 2):
                eight = set(eight).difference(two)
                for three in itertools.combinations(eight, 2):
                    eight = set(eight).difference(three)
                    four = tuple(eight)
                    trans = dict()
                    g = set([tuple(sorted(one)), tuple(sorted(two)), tuple(sorted(three)), tuple(sorted(four))])
                    h = set(one+two+three+four)
                    #print(one, two, three, four)
                    #print(g)
                    #print(h)
                    #_=input()
                    if len(g) != 4 or len(h) != 8:
                        continue
                    for x in [one, two, three, four]:
                        trans[x[0]]=x[1]
                        trans[x[1]]=x[0]
                    #print(one, two, three, four)
                    #print(trans)
                    d = copy.deepcopy(orig_d)
                    counter = 0
                    while any(d.get(z) is None for z in zs) and counter < 100:
                        for i in p[c_loc:]:
                            left, op, right, _, o_dest = i.split()
                            dest = trans[o_dest] if o_dest in trans else o_dest
                            #print(left, d.get(left), op, right, d.get(right), dest)
                            if d.get(dest):
                                continue
                            if d.get(left, None) is not None and d.get(right, None) is not None:
                                if op == "OR":
                                    d[dest] = d[left] | d[right]
                                elif op == "AND":
                                    d[dest] = d[left] & d[right]
                                elif op == "XOR":
                                    d[dest] = d[left] ^ d[right]
                        counter += 1
                        #print(d)
                        #_=input()
                    ans = [str(d.get(z,0)) for z in sorted(zs, reverse=True)]
                    this_ans = int("".join(ans), 2)
                    #print(final, this_ans)
                    if this_ans == final:
                        print("".join(ans))
                        print(",".join(sorted(trans.keys())), this_ans)
                        print("found")
                        print(sorted(trans.keys()))
                        return
            #if ",".join(sorted(trans.keys())) == "z00,z01,z02,z05":
            #    _=input()
            #print(i)
            #print(d)
            #_=input()

    #print(len(p[c_loc:]))


def day25(example=False, override=False):
    """Day 25."""
    day: int | str
    day = ("#####\n"
           ".####\n"
           ".####\n"
           ".####\n"
           ".#.#.\n"
           ".#...\n"
           ".....\n"
           "\n"
           "#####\n"
           "##.##\n"
           ".#.##\n"
           "...##\n"
           "...#.\n"
           "...#.\n"
           ".....\n"
           "\n"
           ".....\n"
           "#....\n"
           "#....\n"
           "#...#\n"
           "#.#.#\n"
           "#.###\n"
           "#####\n"
           "\n"
           ".....\n"
           ".....\n"
           "#.#..\n"
           "###..\n"
           "###.#\n"
           "###.#\n"
           "#####\n"
           "\n"
           ".....\n"
           ".....\n"
           ".....\n"
           "#....\n"
           "#.#..\n"
           "#.#.#\n"
           "#####")
    if not example:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    p = get_input(day, "\n", None, override=override)
    p1 = 0
    locks = []
    keys = []
    this_thing = []
    for l in p:
        if l == "":
            if all(x == 1 for x in this_thing[0]):
                locks.append(np.array(this_thing, dtype=int))
            else:
                keys.append(np.array(this_thing, dtype=int))
            this_thing = []
        else:
            this_thing.append(list(map(lambda x: 1 if x == "#" else 0, list(l))))
    if all(x == 1 for x in this_thing[0]):
        locks.append(np.array(this_thing, dtype=int))
    else:
        keys.append(np.array(this_thing, dtype=int))
    for l in locks:
        for k in keys:
            t = l + k
            if not np.any(t == 2):
                p1 += 1
    print(p1)
