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
# import random
import re
import socket
# import statistics
# import sys
# import time
from os import path

# import cpmpy
# import networkx as nx
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
        return Coordinate(abs(x) for x in self)
    def __mul__(self, other):
        """Multiply a scaler with this coordinate."""
        return Coordinate(x * other for x in self)  # noqa: DOC201
    def __neg__(self):  # noqa: E301
        """Turn the coordinate negative."""
        return Coordinate(-1 * x for x in self)  # noqa: DOC201
    def __add__(self, other):  # noqa: E301
        """Add two coordinates or a coordinate and a tuple."""
        return Coordinate(x + y for x, y in zip(self, other, strict=True))  # noqa: DOC201
    def __sub__(self, other):  # noqa: E301
        """Subtract one coordinate from another."""
        return Coordinate(x - y for x, y in zip(self, other, strict=True))  # noqa: DOC201
    def __lt__(self, other):  # noqa: E301
        """Use to test if coordinate in 2D array."""
        return all(x < y for x, y in zip(self, other, strict=True))  # noqa: DOC201
    def __le__(self, other):  # noqa: E301
        """Use to test if coordinate in 2D array."""
        return all(x <= y for x, y in zip(self, other, strict=True))  # noqa: DOC201
    def __gt__(self, other):  # noqa: E301
        """Use to test if coordinate in 2D array."""
        return all(x > y for x, y, in zip(self, other, strict=True))  # noqa: DOC201
    def __ge__(self, other):  # noqa: E301
        """Use to test if coordinate in 2D array."""
        return all(x >= y for x, y, in zip(self, other, strict=True))  # noqa: DOC201
    def __setitem__(self, key, value):  # noqa: E301
        """Ok, look it really isn't a tuple."""
        self_list = list(self)
        self_list[key] = value
        return Coordinate(tuple(self_list))  # noqa: DOC201
    def manhattan_dist(self, other):  # noqa: E301
        """Calculate the manhattan distance between this coordinate and another."""
        return Coordinate(abs(x - y) for x, y in zip(self, other, strict=True))  # noqa: DOC201


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
    visited = set([node.p()])
    queue = [node]
    paths = 0
    while queue:          # Creating loop to visit each node
        this_node = queue.pop(0)
        for neighbor in ["u","d","l","r"]:
            t=PointObject(*this_node.p())
            t.move(neighbor)
            if t.y in range(graph[0].size) and t.x in range(graph[:,0].size) and graph[t.p()] - 1 == graph[this_node.p()]:
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
    b = np.full((a[0].size, a[:,0].size), -1, dtype=int)
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
                t.append(int(st[:len(st) // 2]))
                t.append(int(st[len(st) // 2:]))
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
    visited = set([node])
    queue = [node]
    corners = 0
    perimeter = 0
    while queue:  # Creating loop to visit each node
        this_node = queue.pop(0)
        this_char = graph[*this_node]
        this_3x3 = graph[this_node[0] - 1:this_node[0] + 2,this_node[1] - 1:this_node[1] + 2]
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
               t[1] in range(graph[:, 0].size) and graph[*t] == graph[*this_node]:
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
    a = get_np_input(day, "\n", splitter=list, dtype=str, override=override)
    total_plots = a[0].size * a[:,0].size
    a = np.pad(a, 1, mode="constant", constant_values=".")
    plot_types = set([a[x] for x in zip(*np.where(a!="."), strict=True)])
    total_visited = set()
    p1 = p2 = 0
    while len(total_visited) < total_plots:
        for t in plot_types:
            plots = set([Coordinate(x) for x in zip(*np.where(a == t), strict=True)]).difference(total_visited)
            while plots:
                temp = plots.pop()
                visited, corners, perimeter = plots_bfs(a, temp)
                # print(a[*temp], "area: ", len(visited), "sides: ", corners, "perimeter: ", perimeter)
                p1 += len(visited) * perimeter
                p2 += len(visited) * corners
                total_visited = total_visited.union(visited)
                plots = set([Coordinate(x) for x in zip(*np.where(a == t), strict=True)]).difference(total_visited)
    print("Part 1: ", p1)
    print("Part 2: ", p2)
