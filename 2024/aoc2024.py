import math
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
import networkx as nx
import numpy as np
import pyglet
import requests
import functools
import heapq
import cpmpy

# Advent of Code
# Never did spend the time to work out how to get oAuth to work so this code expects you to
# manually copy over your session cookie value.
# Using a web browser inspect the cookies when logged into the Advent of Code website.
# Copy the value from the "session" cookie into a text file called "session.txt"

# Constants
_code_path = r'c:\AoC'
_offline = False
_year = 2024


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
            raise Exception("Using the web browser get the session cookie value\nand put it as a string in {}".format(_code_path + r"\session.txt"))  # noqa: W605
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


# Some code experiments in a visualization module instead of using curses.
# class Viz(pyglet.window.Window):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs# )
#
#     def on_draw(self):
#         self.clear()
#         label = pyglet.text.Label("Hello, world",font_name='Times New Roman',font_size=36, x=self.width//2, y=self.height//2, anchor_x='center',anchor_y='center')
#         label.draw() 
#
# def v():
#     #label = pyglet.text.Label("Hello, world",font_name='Times New Roman',font_size=36, anchor_x='center',anchor_y='center')
#     _ = Viz(512, 512, "Test",resizable=False)
#     pyglet.app.run()

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
        return Coordinate(abs(x - y) for x, y in zip(self, other))


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



