import math
import copy
# import curses # pip install windows-curses
import inspect
import pickle
import random
import re
import socket
from os import path
import time, sys

# import functools
import itertools
# import statistics
import collections
import networkx as nx
import numpy as np
# import pyglet
import requests
import functools
import heapq
# import cpmpy

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
        return True
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
        if _check_internet() and False:
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

class Timer (object):
    """
    Performance debugging class. Usage:
       with Timer(<optional label/tag>):
    """
    def __init__(self, label=None, on=True):
        self.label = label
        self.on=on

    def __enter__(self):
        self.tic = time.perf_counter()
        return(self)

    def __exit__(self, type, value, traceback):
        elapsed = time.perf_counter() - self.tic
        if self.on==True:
            elapsed_us = elapsed*1000000
            print("Finished %s after %.0f us\n"%(self.label, elapsed_us))
        self.elapsed = elapsed

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


def day1_original(example=False):
    """So it begins."""
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
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
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
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
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
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
        print("")
        print(lvl)
        print(a)
        if a[-1] > 3:
            continue
        if a[0] < -3:
            continue
        if 0 in a:
            continue
        b=np.where(a<0)
        if np.where(a<0)[0].size == a.size:
            print("safe neg")
            p1 += 1
        if np.where(a>0)[0].size == a.size:
            print("safe pos")
            p1+=1
    print(p1)


def safe_level_orig(lvl):
    """Check if the level is safe."""
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
    if np.where(a > 0)[0].size == a.size:
        return True
    return False


def day2_part2_original(example=False):
    """Day 2."""
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
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
    print(np.diff(np_array))
    pos_bad_locations = np.where(np.isin(np.diff(np_array), [1, 2, 3], invert=True))[0]
    neg_bad_locations = np.where(np.isin(np.diff(np_array), [-1, -2, -3], invert=True))[0]
    


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
        _=input()
    return safe, dampener_safe


def day2(example=False, override=False):
    """Day 2."""
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
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


def mul(a,b):
    return a*b


def day3_eval(example=True, override=False):
    """
    Day 3.

    Args:
        example (bool): Use the example input.
        override (boot): Override the stored get_input data.
    """
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    if example:
        day = """xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))"""
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    r=re.compile("mul\([0-9]*,[0-9]*\)|do\(\)|don't\(\)")
    do=True
    for x in p:
        l = r.findall(x)
        print(l)
        for i in l:
            if i == "do()":
                do=True
                continue
            if i == "don't()":
                do=False
                continue
            p1 += eval(i)
            if do:
                p2 += eval(i)
    print(p1)
    print(p2)


def day3(example=True, override=False):
    """
    Day 3.

    Args:
        example (bool): Use the example input.
        override (boot): Override the stored get_input data.
    """
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    if example:
        day = """xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))"""
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    r=re.compile("mul\([0-9]*,[0-9]*\)|do\(\)|don't\(\)")
    do=True
    for x in p:
        l = r.findall(x)
        print(l)
        for i in l:
            if i == "do()":
                do=True
                continue
            if i == "don't()":
                do=False
                continue
            a=int(i.split("(")[1].split(",")[0])
            b=int(i.split(",")[1].split(")")[0])
            print(i, a, b)
            p1 += (a*b)
            if do:
                p2 += (a*b)
    print(p1)
    print(p2)


def day4(example=False, override=False):
    """Day 4."""
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    if example:
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
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    the_map= []
    for l in p:
        the_map.append(list(l))
    the_xs=[]
    # Part 1
    # Find the X's
    for y,l in enumerate(p):
        for x,c in enumerate(l):
            if c == "X":
                the_xs.append(Point_Object(y,x))
    # Check for XMAS
    for x in the_xs:
        for md in ["u","d","l","r","ur","ul","dl","dr"]:
            t = Point_Object(*x.p())
            t.move(md)
            if t.y not in range(len(the_map)) or t.x not in range(len(the_map[0])) or the_map[t.y][t.x] != "M":
                continue
            t.move(md)
            if t.y not in range(len(the_map)) or t.x not in range(len(the_map[0])) or the_map[t.y][t.x] != "A":
                continue
            t.move(md)
            if t.y not in range(len(the_map)) or t.x not in range(len(the_map[0])) or the_map[t.y][t.x] != "S":
                continue
            p1+=1
    # Part 2
    # Find the A's
    the_as = []
    for y,l in enumerate(p):
        for x,c in enumerate(l):
            if c == "A":
                the_as.append(Point_Object(y,x))
    # Check for MAS
    for a in the_as:
        ur = Point_Object(*a.p())
        ur.move("ur")
        ul = Point_Object(*a.p())
        ul.move("ul")
        dr = Point_Object(*a.p())
        dr.move("dr")
        dl = Point_Object(*a.p())
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
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    if example:
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
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    r = []
    u = []
    for l in p:
        if "|" in l:
            a, b = l.split("|")
            r.append((int(a), int(b)))
        if "," in l:
            u.append(list(map(int, l.split(","))))
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


global rules


def _page_sort(a, b):
    global rules
    if (a, b) in rules:
        return -1
    elif (b, a) in rules:
        return 1
    else:
        return 0


def day5_sort(example=False, override=False):
    """Day 5."""
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    if example:
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
    p1 = p2 = 0
    p = get_input(day, "\n", None, override=override)
    global rules
    rules = set()
    u = []
    for l in p:
        if "|" in l:
            rules.add(tuple(map(int, l.split("|"))))
        if "," in l:
            u.append(list(map(int, l.split(","))))
    for x in u:
        n = sorted(x, key=functools.cmp_to_key(_page_sort))
        if n == x:
            p1 += x[len(x) // 2]
        else:
            p2 += n[len(n) // 2]
    print(p1)
    print(p2)
