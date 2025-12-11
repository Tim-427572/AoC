"""
Advent of Code.

Never did spend the time to work out how to get oAuth to work so this code expects you to
manually copy over your session cookie value.
Using a web browser inspect the cookies when logged into the Advent of Code website.
Copy the value from the "session" cookie into a text file called "session.txt"
"""  # noqa: CPY001

import copy
# import cpmpy as cp
# import curses  # pip install windows-curses
import dill
import inspect
import itertools
import logging
import math
import networkx as nx
import numpy as np
import pathlib
import re
import requests  # type: ignore[import-untyped]
import socket
import sys

from argparse import ArgumentParser
from collections import defaultdict
from os import path
from shapely import geometry

# Constants
_NAME = pathlib.Path(__file__).stem
_CODE_PATH = r"c:\AoC"
_YEAR = 2025
_OFFLINE = False

# Create and configure logger
logging.basicConfig(level=logging.INFO,
                    format="%(message)s",
                    handlers=[logging.FileHandler(filename=f"{_NAME}.log", mode="a"),
                              logging.StreamHandler()])
log = logging.getLogger()
# log.propagate = False


def _check_internet(host="8.8.8.8", port=853, timeout=2):
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
    if _OFFLINE:
        with open(_CODE_PATH + r"\{}\day{}.txt".format(_YEAR, day)) as file_handler:  # noqa: UP032, FURB101, PTH123
            data_list = file_handler.read().split(seperator)
    elif isinstance(day, str):  # An example string
        data_list = day.split(seperator)
    else:
        if not path.exists(_CODE_PATH + "/session.txt"):  # noqa: PTH110
            raise Exception("Using the web browser get the session cookie value\nand put it as a string in {}".format(_CODE_PATH + r"\session.txt"))  # noqa: W605, EM103, TRY002
        with open(_CODE_PATH + "/session.txt", 'r') as session_file:  # noqa: Q000, UP015, FURB101, PTH123
            session = session_file.read()
        # Check to see if behind the firewall.
        if _check_internet():  # noqa: SIM108
            proxy_dict = {}
        else:
            proxy_dict = {"http": "proxy-dmz.intel.com:911",
                          "https": "proxy-dmz.intel.com:912"}
        header = {"Cookie": "session={:s}".format(session.rstrip("\n"))}
        with requests.Session() as session:
            resp = session.get("https://adventofcode.com/{}/day/{}/input".format(_YEAR, day), headers = header, proxies = proxy_dict)  # noqa: E251, UP032
            _ = resp.text.strip("\n")
            if resp.ok:
                if seperator in [None, ""]:  # noqa: PLR6201, SIM108
                    data_list = [resp.text]
                else:
                    data_list = resp.text.split(seperator)
            else:
                log.info("Warning website error")
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
    # if path.exists(_CODE_PATH + r"\{}\input.p".format(_YEAR)):  # noqa: UP032, PTH110, SIM108
    if path.exists(_CODE_PATH + r"\{}\input.d".format(_YEAR)):  # noqa: UP032, PTH110, SIM108
        puzzle_dict = dill.load(open(_CODE_PATH + r"\{}\input.d".format(_YEAR), "rb"))  # noqa: UP032, SIM115, PTH123, S301
        # puzzle_dict = pickle.load(open(_CODE_PATH + r"\{}\input.p".format(_YEAR), "rb"))  # noqa: UP032, SIM115, PTH123, S301
    else:  # No pickle file, will need to make a new one.
        puzzle_dict = {}

    puzzle_input = puzzle_dict.get(day)

    if puzzle_input is None or override is True:
        puzzle_input = _pull_puzzle_input(day, seperator, cast)
        if isinstance(day, int):  # only save the full puzzle data to the pickle file.
            puzzle_dict[day] = puzzle_input
            dill.dump(puzzle_dict, open(_CODE_PATH + r"\{}\input.d".format(_YEAR), "wb"))  # noqa: UP032, SIM115, PTH123, S301
            # pickle.dump(puzzle_dict, open(_CODE_PATH + r"\{}\input.p".format(_YEAR), "wb"))  # noqa: UP032, SIM115, PTH123, S301
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


def print_np(array, seperator=""):
    """Small script to print a numpy array to the console visually similar to the puzzles in AoC."""
    if array.dtype == np.dtype("<U1"):
        for row in array:
            log.info("".join(row))
    else:
        for row in array:
            log.info(np.array2string(row, separator=seperator, max_line_width=600)[1:-1])


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
        log.info(f"({self.y},{self.x}) {self.contains}")
        if self.direction:
            log.info(f"direction: {self.direction}")
        if self.up:
            log.info(f"up: ({self.up[0].y}, {self.up[0].x}) {self.up[1]}")
        if self.down:
            log.info(f"down: ({self.down[0].y}, {self.down[0].x}) {self.down[1]}")
        if self.left:
            log.info(f"left: ({self.left[0].y}, {self.left[0].x}) {self.left[1]}")
        if self.right:
            log.info(f"right: ({self.right[0].y}, {self.right[0].x}) {self.right[1]}")


# A thing that isn't really a tuple which makes tracking 2D points easier.
class Coordinate(tuple):  # noqa: SLOT001
    """
    Like a tuple but not.

    Used to store 2D position but still allow hashing and (x,y) notation which I like.
    """

    def __abs__(self):
        """Return absolute value for each part."""
        return Coordinate(abs(x) for x in self)
    def __mul__(self, other):  # noqa: W291, E301
        """Multiply a scaler with this coordinate."""  # noqa: DOC201
        return Coordinate(x * other for x in self)
    def __neg__(self):  # noqa: W291, E301
        """Turn the coordinate negative."""  # noqa: DOC201
        return Coordinate(-1 * x for x in self)
    def __add__(self, other):  # noqa: W291, E301
        """Add two coordinates or a coordinate and a tuple."""  # noqa: DOC201
        return Coordinate(x + y for x, y in zip(self, other, strict=True))
    def __sub__(self, other):  # noqa: W291, E301
        """Subtract one coordinate from another."""  # noqa: DOC201
        return Coordinate(x - y for x, y in zip(self, other, strict=True))
    def __lt__(self, other):  # noqa: W291, E301
        """Use to test if coordinate in 2D array."""  # noqa: DOC201
        return all(x < y for x, y in zip(self, other, strict=True))
    def __le__(self, other):  # noqa: W291, E301
        """Use to test if coordinate in 2D array."""  # noqa: DOC201
        return all(x <= y for x, y in zip(self, other, strict=True))
    def __gt__(self, other):  # noqa: W291, E301
        """Use to test if coordinate in 2D array."""  # noqa: DOC201
        return all(x > y for x, y, in zip(self, other, strict=True))
    def __ge__(self, other):  # noqa: W291, E301
        """Use to test if coordinate in 2D array."""  # noqa: DOC201
        return all(x >= y for x, y, in zip(self, other, strict=True))
    def __setitem__(self, key, value):  # noqa: W291, E301
        """Ok, look it really isn't a tuple."""  # noqa: DOC201
        self_list = list(self)
        self_list[key] = value
        return Coordinate(tuple(self_list))
    def manhattan_dist(self, other):  # noqa: W291, E301
        """Calculate the manhattan distance between this coordinate and another."""  # noqa: DOC201
        return Coordinate(abs(x - y) for x, y in zip(self, other, strict=True))
    def euclidean_dist(self, other):  # noqa: W291, E301
        """Calculate the euclidean distance between this coordinate and anotehr."""  # noqa: DOC201
        return math.sqrt(sum((x - y)**2 for x, y in zip(self, other, strict=True)))


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
        log.info(node)
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor)


def bfs(graph, node):  # Example function for BFS
    """BFS search."""
    visited = set()
    queue = [node]

    while queue:          # Creating loop to visit each node
        this_node = queue.pop(0)
        log.info(this_node)

    for neighbor in graph[this_node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)


def day1(example=False, override=False, **kwargs):
    """So it begins."""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "L68\nL30\nR48\nL5\nR60\nL55\nL1\nL99\nR14\nL82"
    rotations = get_input(day, "\n", override=override)
    # log.info(rotations)
    p1_password = 0
    p2_password = 0
    dial = 50
    for rotation in rotations:
        num = int(rotation[1:])
        old_dial = dial
        # Full rotations
        p2_password += num // 100
        num %= 100
        if "R" in rotation:
            dial += num
            if dial > 99:
                dial -= 100
                p2_password += 1 if dial != 0 else 0
        else:
            dial -= num
            if dial < 0:
                dial += 100
                if old_dial != 0:
                    p2_password += 1
        if dial == 0:
            p1_password += 1
            p2_password += 1
        # log.info("%s %d->%d password=%d" % (rotation.replace("L", "-").replace("R", "+"), old_dial, dial, password))
    log.info("Part 1: %s", p1_password)
    log.info("Part 2: %s", p2_password)


def day2(example=False, override=False, **kwargs):
    """Valid IDs."""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,"
               "38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124")
    p = get_input(day, ",", override=override)
    part1 = set()
    part2 = set()
    for r in p:
        s, e = map(int, r.split("-"))
        for x in range(s, e + 1):
            y = str(x)
            for z in range(1, (len(y) // 2) + 1):
                if len(y) % z != 0:
                    continue
                h = len(y) // z
                if len(re.findall(y[:z], y)) == h:
                    # log.info(y[:z], y, h, re.findall(y[:z], y))
                    part2.add(x)
                if len(re.findall(y[:z], y)) == h and h == 2:
                    part1.add(x)
    log.info("Part 1: %s", sum(part1))
    log.info("Part 2: %s", sum(part2))


def day3(digits="2", example=False, override=False, **kwargs):
    """How many batteries?"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "987654321111111\n811111111111119\n234234234234278\n818181911112111\n"
    banks = get_input(day, "\n", cast=lambda x: [int(y) for y in x], override=override)
    total_joltage = 0
    for bank in banks:
        joltage = 0
        for batteries in range(int(digits), 0, -1):
            for rating in range(9, 0, -1):
                possible_max_rating = min(rating, max(bank))
                if possible_max_rating not in bank:
                    continue
                if len(bank) - bank.index(possible_max_rating) >= batteries:
                    break
            joltage = (joltage * 10) + possible_max_rating
            bank = bank[bank.index(possible_max_rating) + 1:]  # Shrink the set of batteries. # noqa: PLW2901
        total_joltage += joltage
    log.info(f"Total joltage for {digits} digits: {total_joltage}")


def day4(example=False, override=False, **kwargs):
    """???"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "..@@.@@@@.\n@@@.@.@.@@\n@@@@@.@.@@\n@.@@@@..@.\n@@.@@@@.@@\n.@@@@@@@.@\n.@.@.@.@@@\n@.@@@.@@@@\n.@@@@@@@@.\n@.@.@@@.@."
    p = get_np_input(day, "\n", cast=str, splitter=list, override=override)
    p = np.pad(p, 1, mode="constant", constant_values=".")
    # print_np(p)
    p1 = None
    p2_set = set()
    removed = count = 0
    while True:
        if removed == len(p2_set):
            count += 1  # Do a couple extra loops just to be sure we are done.
            if count >= 4:
                break
        else:
            removed = len(p2_set)
        # Remove the paper rolls.
        for x in p2_set:
            p[x] = "."
        for r in range(1, p[0, :].size):
            for c in range(1, p[:, 0].size):
                a = p[r - 1: r + 2, c - 1: c + 2]
                if p[r, c] == "@" and np.count_nonzero(a[a == "@"]) < 5:
                    p2_set.add((r, c))
        if p1 is None:  # Part 1 is just the first time through the loop.
            p1 = len(p2_set)
    log.info(f"Part 1: {p1}")
    log.info(f"Part 2: {removed}")


def range_limit_sort(a, b):
    """Custom sort function."""  # noqa: DOC201
    return -1 if a.split("-")[1] < b.split("-") else 1


def day5(example=False, override=False, **kwargs):  # noqa: RET503
    """The fresh maker!"""  # noqa: DOC201
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "3-5\n10-14\n16-20\n12-18\n\n1\n5\n8\n11\n17\n32"
    puzzle = get_input(day, "\n", cast=None, override=override)
    part_1 = part_2 = 0
    fresh_list = [list(map(int, fresh_range.split("-"))) for fresh_range in puzzle[:puzzle.index("")]]
    ingredients = [int(x) for x in puzzle[puzzle.index("") + 1:]]
    log.debug(sorted(fresh_list))
    size = 0
    while size != len(fresh_list):  # Condense the freshness ID ranges until we can't no more...
        size = len(fresh_list)
        fresh_list = sorted(fresh_list)
        condensed_list = [fresh_list.pop(0)]
        while fresh_list:
            first = condensed_list.pop()
            second = fresh_list.pop(0)
            if first[1] >= second[1] and second[0] >= first[0]:  # Second in frst
                condensed_list.append(first)
            elif first[1] >= second[0]:  # overlap between first and second.
                condensed_list.append([first[0], second[1]])
            else:  # No overlap.
                condensed_list += [first, second]
        fresh_list = copy.deepcopy(condensed_list)
        log.debug(fresh_list)
    # Now solve the puzzle.
    for fresh in fresh_list:
        part_2 += fresh[1] - fresh[0] + 1
        for ingredient in ingredients:
            if fresh[1] >= ingredient >= fresh[0]:
                part_1 += 1
    log.info(f"Part 1: {part_1}")
    log.info(f"Part 2: {part_2}")


def day6_part1(example=False, override=False, **kwargs):
    """Elf spreadsheets!"""  # noqa: DOC501
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = """123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +"""
    p = get_np_input(day, seperator="\n", splitter=lambda x: x.split(), override=override)
    log.info(p)
    p1 = p2 = 0
    for i in range(p.shape[1]):
        t = p[:, i]
        if t[-1] == "+":
            p1 += sum(t[:-1].astype(np.int64))
        elif t[-1] == "*":
            p1 += np.prod(t[:-1].astype(np.int64))
        else:
            log.info(t)
            raise Exception("What?")  # noqa: EM101, TRY002
    log.info(f"Part 1: {p1}")
    log.info(f"Part 2: {p2}")


def day6(example=False, override=False, **kwargs):
    """Stupid spreadsheets!"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("123 328  51 64 \n"
               " 45 64  387 23 \n"
               "  6 98  215 314\n"
               "*   +   *   +  ")
    p = get_np_input(day, seperator="\n", splitter=list, override=override)
    p1 = 0
    r = []
    op_func = np.sum
    for t in p.T:
        op = t[-1]
        t = np.delete(t, -1)  # noqa: PLW2901
        if op != " ":  # New set of numbers, do math on the old ones.
            p1 += op_func(r, dtype=np.int64)
            op_func = np.sum if op == "+" else np.prod
            r = []
        if any(t != " "):
            r.append(int("".join(t[t != " "])))
    p1 += op_func(r, dtype=np.int64)  # Add in the last column of values.
    log.info(f"Part 2: {p1}")


def day7(example=False, override=False, **kwargs):
    """???."""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = (".......S.......\n"
               "...............\n"
               ".......^.......\n"
               "...............\n"
               "......^.^......\n"
               "...............\n"
               ".....^.^.^.....\n"
               "...............\n"
               "....^.^...^....\n"
               "...............\n"
               "...^.^...^.^...\n"
               "...............\n"
               "..^...^.....^..\n"
               "...............\n"
               ".^.^.^.^.^...^.\n"
               "...............")
    # Turn the map into an array where splitters are -1 and beams are numbers > 0
    p = get_np_input(day=day, seperator="\n", cast=None, splitter=lambda x: [{".": 0, "S": 1, "^": -1}[y] for y in x], dtype=np.int64, override=override)
    p1 = 0
    p = np.pad(p, 1, mode="constant", constant_values=0)  # Pad the array just in case.
    for above, below in itertools.pairwise(p):
        for beam_loc in np.where(above > 0)[0]:
            if below[beam_loc] == -1:  # Splitter encountered
                p1 += 1
                below[beam_loc + 1] += above[beam_loc]   # This only works because there are no adjacent splitters in the puzzle data.
                below[beam_loc - 1] += above[beam_loc]
            else:
                below[beam_loc] += above[beam_loc]
    log.info(f"Part 1: {p1}")
    log.info(f"Part 2: {np.sum(p[-1])}")


def day8(example=False, override=False, **kwargs):
    """Aziz, LIGHT!"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("162,817,812\n57,618,57\n906,360,560\n592,479,940\n352,342,300\n466,668,158\n542,29,236\n431,825,988\n739,650,466\n52,470,668\n"
               "216,146,977\n819,987,18\n117,168,530\n805,96,715\n346,949,466\n970,615,88\n941,993,340\n862,61,35\n984,92,344\n425,690,689")
    p = get_input(day=day, seperator="\n", cast=lambda x: Coordinate(map(int, x.split(","))), override=override)
    distances = {}
    g = nx.Graph()
    g.add_nodes_from(p)
    # Calculate distances
    for x, y in itertools.combinations(p, 2):
        euclidean_dist = x.euclidean_dist(y)
        if euclidean_dist in distances:
            log.error("Why are there duplicates?")
        else:
            distances[euclidean_dist] = (x, y)
    # Hook up the lights:
    for i, distance in enumerate(sorted(distances.keys()), start=1):
        x, y = distances[distance]
        if x not in g[y]:
            g.add_edge(x, y)
        if (example and i == 10) or (not example and i == 1000):
            p1 = np.prod([len(x) for x in sorted(nx.connected_components(g), key=len, reverse=True)[:3]])
            log.info(f"Part 1: {p1}")
        if nx.number_of_isolates(g) == 0:
            log.info(f"Part 2: {x[0] * y[0]}")
            break


def day9(example=False, override=False, **kwargs):
    """Red tiles."""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "7,1\n11,1\n11,7\n9,7\n9,5\n2,5\n2,3\n7,3"
    p = get_input(day=day, seperator="\n", cast=lambda x: Coordinate(map(int, x.split(","))), override=override)
    distances = defaultdict(list)
    green_tiles = geometry.Polygon(p)
    # Calculate areas, this seems a lot like day 8 which is kind of boring.
    for x, y in itertools.combinations(p, 2):
        manhattan_dist = x.manhattan_dist(y)
        area = (manhattan_dist[0] + 1) * (manhattan_dist[1] + 1)
        distances[area].append((x, y))
    sorted_areas = sorted(distances.keys(), reverse=True)
    log.info(f"Part 1: {sorted_areas[0]}")
    # Part 2, check if rectangle is inside the green tile shape.
    for area in sorted_areas:
        for x, y in distances[area]:
            rectangle = geometry.Polygon([x, (x[0], y[1]), y, (y[0], x[1])])
            if rectangle.covered_by(green_tiles):
                log.debug("Inside %s %s %s", area, x, y)
                log.info(f"Part 2: {area}")
                return
            log.debug("Outside %s %s %s", area, x, y)
    return


def _button_to_int(s):
    """Take the puzzle comma seperated list and turn it into an int."""
    t = [1 << int(x) for x in s.split(",")]
    return np.bitwise_or.reduce(t)


def day10_1(example=False, override=False, **kwargs):
    """There are three lights!!!"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}\n"
               "[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}\n"
               "[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}")
    p = get_input(day=day, seperator="\n", cast=None, override=override)
    p1 = 0
    for machine in p:
        log.debug(machine)
        patterns = set()
        result = int(re.sub(r".|#", lambda x: {".": "0", "#": "1"}[x.group(0)], re.search(r"\[([.#]*)\]", machine).group(1))[::-1], 2)
        buttons = [_button_to_int(x) for x in re.findall(r"\(([0-9,]*)\)", machine)]
        # load the buttons in as a starting point
        for button in buttons:  # noqa: FURB142
            patterns.add(frozenset({button}))
        # Start a kind of BFS looking for the shortest sequence
        while True:
            new = set()
            for seq in patterns:
                log.debug(f"{result} == {np.bitwise_xor.reduce(list(seq))} {seq}")
                if result == np.bitwise_xor.reduce(list(seq)):
                    log.info(f"Solution {result} from {list(seq)}")
                    p1 += len(seq)
                    break
                for button in buttons:
                    new_seq = frozenset([*seq, button])
                    if new_seq not in patterns:
                        new.add(new_seq)
            else:
                patterns = new
                # _ = input()
                continue
            break
    log.info(f"Part 1: {p1}")


def _button_to_coord(s, dimension):
    """Take the puzzle data and turn it into a Coordinate object."""
    t = [0] * dimension
    for i in s.split(","):
        t[int(i)] = 1
    return Coordinate(t)


def day10(example=False, override=False, **kwargs):
    """There are three lights!!!"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}\n"
               "[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}\n"
               "[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}")
    p = get_input(day=day, seperator="\n", cast=None, override=override)
    p1 = p2 = 0
    for machine in p:
        log.debug(machine)
        result = Coordinate(map(int, re.search(r"\{([0-9,]*)\}", machine).group(1).split(",")))
        log.debug(result)
        buttons = [_button_to_coord(x, len(result)) for x in re.findall(r"\(([0-9,]*)\)", machine)]
        log.debug(buttons)
        seq_dict = {}
        min_presses = None
        for button in buttons:
            seq_dict[button] = max(result[i] * button[i] for i in range(len(result)))
        log.debug(f"seq_dict: {seq_dict}")
        # load the buttons in each being pushed as many times as the max in the desired joltage.
        patterns = {frozenset(seq_dict.items())}
        for p in patterns:
            log.debug(p)
        # Start a kind of BFS looking for the shortest sequence but walking backwards.
        while patterns:
            new = set()
            for froz_seq in patterns:
                temp = Coordinate([0] * len(result))
                seq = dict(froz_seq)
                for k, v in seq.items():
                    temp += k * v
                log.debug(f"{result} == {temp} for {seq}")
                if result == temp:
                    # log.info(f"Solution {result} at {sum(seq.values())} from {seq}")
                    min_presses = min(min_presses, sum(seq.values())) if min_presses is not None else sum(seq.values())
                    # _ = input()
                    # p2 += sum(seq.values())
                    # break
                if any(x < 0 for x in temp - result):  # Too few button presses.
                    continue
                for button in buttons:
                    new_seq = copy.deepcopy(seq)
                    new_seq[button] -= 1
                    if new_seq[button] < 0:  # Too far, abort this sequence.
                        continue
                    new_seq = frozenset(new_seq.items())
                    if new_seq not in patterns and new_seq not in new:
                        new.add(new_seq)
            else:  # noqa:  PLW0120
                patterns = copy.deepcopy(new)
                continue
            break
        log.info(f"Solution {result} at {min_presses}")
        p2 += min_presses

    log.info(f"Part 1: {p1}")
    log.info(f"Part 2: {p2}")


def day11(start="you", example=False, override=False, **kwargs):
    """Server rack madness!"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example and start == "you":
        day = ("aaa: you hhh\nyou: bbb ccc\nbbb: ddd eee\nccc: ddd eee fff\nddd: ggg\n"
               "eee: out\nfff: out\nggg: out\nhhh: ccc fff iii\niii: out")
    if example and start == "svr":
        day = ("svr: aaa bbb\naaa: fft\nfft: ccc\nbbb: tty\ntty: ccc\nccc: ddd eee\n"
               "ddd: hub\nhub: fff\neee: dac\ndac: fff\nfff: ggg hhh\nggg: out\nhhh: out")
    p = get_input(day=day, seperator="\n", cast=None, override=override)
    g = nx.DiGraph()
    for x in p:
        dev, outputs = x.split(":")
        for out in outputs.split():
            g.add_edge(dev, out)
    log.debug(g)
    topological_g = nx.topological_sort(g)
    path_dict = defaultdict(int)
    path_dict[start] = 1
    for node in topological_g:
        log.debug(f"Working on {node}")
        if node in {"fft", "dac"} and start == "svr":
            temp = path_dict[node]
            path_dict.clear()
            path_dict[node] = temp
        for _, edge in g.out_edges(node):
            log.debug(f"Processing edge {edge}")
            path_dict[edge] += path_dict[node]
            log.debug("%s %s", path_dict[node], path_dict[edge])
        log.debug("%s %s", node, path_dict)
    log.info(f"There are {path_dict['out']} paths from '{start}' to 'out'")


# Template
def day(example=False, override=False, **kwargs):
    """???."""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = """"""
    p = get_input(day=day, seperator="\n", cast=None, override=override)
    # p = get_np_input(day=day, seperator="\n", cast=None, splitter=list, dtype=str, override=override)
    log.info(p)
    p1 = p2 = 0
    # for x in p:
    #     p1 += x
    log.info(f"Part 1: {p1}")
    log.info(f"Part 2: {p2}")


def main(argv=None):
    """Command line interface."""
    argparser = ArgumentParser(prog=_NAME)
    argparser.add_argument("--day", type=int, default=1, help="AoC day to run.")
    argparser.add_argument("--example", dest="example", action="store_true", default=False,
                           help="Run the code against the puzzle example.")
    argparser.add_argument("--override", dest="override", action="store_true", default=False,
                           help="Override the stored puzzle input data.")
    argparser.add_argument("--debug", dest="debug", action="store_true", default=False,
                           help="Set debug logging level.")
    args, unknown = argparser.parse_known_args(argv)
    kwargs = {x.removeprefix("--").split("=")[0]: x.split("=")[1] for x in unknown}
    if args.debug:
        log.setLevel(logging.DEBUG)
    if hasattr(sys.modules[__name__], f"day{args.day}"):
        day = getattr(sys.modules[__name__], f"day{args.day}")
        day(**vars(args), **kwargs)
    else:
        log.info("%s.py does not have a function called day%s", _NAME, args.day)


if __name__ == "__main__":
    sys.exit(main())
