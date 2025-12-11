"""
Advent of Code.

Never did spend the time to work out how to get oAuth to work so this code expects you to
manually copy over your session cookie value.
Using a web browser inspect the cookies when logged into the Advent of Code website.
Copy the value from the "session" cookie into a text file called "session.txt"
"""  # noqa: CPY001

import hashlib
import inspect
import logging
import math
import pathlib
import re
import socket
import sys
# from functools import lru_cache
from argparse import ArgumentParser
from collections import defaultdict
from os import path
from string import ascii_lowercase

import dill
import numpy as np
import requests  # type: ignore[import-untyped]

# Constants
_NAME = pathlib.Path(__file__).stem
_CODE_PATH = r"c:\AoC"
_YEAR = 2016
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
    if path.exists(_CODE_PATH + r"\{}\input.d".format(_YEAR)):  # noqa: UP032, PTH110, SIM108
        puzzle_dict = dill.load(open(_CODE_PATH + r"\{}\input.d".format(_YEAR), "rb"))  # noqa: UP032, SIM115, PTH123, S301
    else:  # No pickle file, will need to make a new one.
        puzzle_dict = {}

    puzzle_input = puzzle_dict.get(day)

    if puzzle_input is None or override is True:
        puzzle_input = _pull_puzzle_input(day, seperator, cast)
        if isinstance(day, int):  # only save the full puzzle data to the pickle file.
            puzzle_dict[day] = puzzle_input
            dill.dump(puzzle_dict, open(_CODE_PATH + r"\{}\input.d".format(_YEAR), "wb"))  # noqa: UP032, SIM115, PTH123, S301
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
move_dict = { "u": Coordinate((-1,  0)),  "n": Coordinate((-1,  0)),         "up": Coordinate((-1,  0)),      "north": Coordinate((-1,  0)),  # noqa: E241, E201
              "d": Coordinate(( 1,  0)),  "s": Coordinate(( 1,  0)),       "down": Coordinate(( 1,  0)),      "south": Coordinate(( 1,  0)),  # noqa: E241, E201
              "r": Coordinate(( 0,  1)),  "e": Coordinate(( 0,  1)),      "right": Coordinate(( 0,  1)),       "east": Coordinate(( 0,  1)),  # noqa: E241, E201
              "l": Coordinate(( 0, -1)),  "w": Coordinate(( 0, -1)),       "left": Coordinate(( 0, -1)),       "west": Coordinate(( 0, -1)),  # noqa: E241, E201
             "ur": Coordinate((-1,  1)), "ne": Coordinate((-1,  1)),   "up-right": Coordinate((-1,  1)), "north-east": Coordinate((-1,  1)),  # noqa: E241, E201
             "dr": Coordinate(( 1,  1)), "se": Coordinate(( 1,  1)), "down-right": Coordinate(( 1,  1)), "south-east": Coordinate(( 1,  1)),  # noqa: E241, E201
             "ul": Coordinate((-1, -1)), "nw": Coordinate((-1, -1)),    "up-left": Coordinate((-1, -1)), "north-west": Coordinate((-1, -1)),  # noqa: E241, E201
             "dl": Coordinate(( 1, -1)), "sw": Coordinate(( 1, -1)),  "down-left": Coordinate(( 1, -1)), "south-west": Coordinate(( 1, -1)),  # noqa: E241, E201
              "^": Coordinate((-1,  0)),  "v": Coordinate(( 1,  0)),          ">": Coordinate(( 0,  1)),          "<": Coordinate(( 0, -1))}  # noqa: E241, E201

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


def day1(example=False, override=False, **kwargs):  # noqa: ARG001
    """Damn Easter Bunny!"""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "R5, L5, R5, R3"
        # day = "R2, R2, R2"
        day = "R8, R4, R4, R8"
    instructions = get_input(day=day, seperator=", ", cast=lambda x: (x[0].lower(), int(x[1:])), override=override)
    current_direction = "north"
    position = Coordinate((0, 0))
    locations = {position}
    part_2 = None
    for turn, move in instructions:
        current_direction = turn_dict[turn][current_direction]
        for _ in range(move):
            position += move_dict[current_direction]
            if position in locations and part_2 is None:
                part_2 = position
            locations.add(position)
    log.info(f"Part 1, you are {sum(position.manhattan_dist((0, 0)))} blocks away")
    log.info(f"Part 2, the first location you visit twice was {sum(part_2.manhattan_dist((0, 0)))} blocks away")  # type: ignore[union-attr]


def day2(part: str | int = 1, example: bool = False, override: bool = False, **kwargs):  # noqa: ARG001
    """I need to pee!"""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "ULL\nRRDDD\nLURDL\nUUUUD"
    instructions = get_input(day=day, seperator="\n", cast=str.lower, override=override)
    # move = {"U": (-1, 0), "D":(1,0), "L":(0,-1), "R":(0,1)}
    if int(part) == 1:  # Part 1 keypad and start position
        position = np.array((1, 1))
        keypad = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    else:  # Part 2 keypad and start position
        position = np.array((2, 0))
        keypad = np.array([[0, 0, 1, 0, 0], [0, 2, 3, 4, 0], [5, 6, 7, 8, 9], [0, "A", "B", "C", 0], [0, 0, "D", 0, 0]])
    valid = []
    for i in range(5):
        for j in range(5):
            if keypad[(i, j)] not in {"0", 0}:  # noqa: RUF031
                valid.append((i, j))  # noqa: PERF401
    code = ""
    for instruction in instructions:
        for letter in instruction:
            new_position = position + np.array(move_dict[letter])
            if tuple(new_position) in valid:
                position = new_position
        code += f"{keypad[tuple(position)]}"
        log.debug(f"{letter} {tuple(position)} {keypad[tuple(position)]}")
    log.info(f"Part {part}: The code is {code}")


def day3(example=False, override=True, **kwargs):  # noqa: ARG001
    """Triangles."""
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = " 5 10 25\n 5 26 25\n2 2 3"
    day = 3
    design_docs = get_input(day=day, seperator="\n", cast=None, override=override)
    possible_triangles = 0
    for triangle in design_docs:
        dimensions = list(map(int, triangle.split()))
        if dimensions[0] + dimensions[1] > dimensions[2] and \
           dimensions[0] + dimensions[2] > dimensions[1] and \
           dimensions[1] + dimensions[2] > dimensions[0]:
            possible_triangles += 1
    log.info(f"Part 1 the possible triangles are {possible_triangles}")
    possible_triangles = 0
    design_docs = list(design_docs)
    while len(design_docs) > 0:
        threes = np.zeros([3, 3], int)
        for i in range(3):
            row = list(map(int, design_docs.pop().split()))
            threes[:, i] = row
        for i in range(3):
            dimensions = list(threes[i])
            if dimensions[0] + dimensions[1] > dimensions[2] and \
               dimensions[0] + dimensions[2] > dimensions[1] and \
               dimensions[1] + dimensions[2] > dimensions[0]:
                possible_triangles += 1
    log.info(f"Part 2 the possible triangles are {possible_triangles}")


def _room_decode(name):
    name_id, checksum = name.split("[")
    checksum = checksum.strip("]")
    sector_id = name_id.split("-")[-1]
    name_letters = name_id.strip(sector_id).strip("-")
    letter_freq: dict[int, list[str]] = defaultdict(list)
    for c in ascii_lowercase:
        freq = name_letters.count(c)
        if freq > 0:
            letter_freq[freq].append(c)
    return (name_letters, letter_freq, sector_id, checksum)


def _decrypt(name, sector_id):
    name_list = name.split("-")
    decrypted = ""
    for word in name_list:
        result = ""
        for l in word:  # noqa: E741
            l_num = ord(l) - 0x60
            l_num += sector_id
            l_num %= 26
            l_num = 26 if l_num == 0 else l_num
            result += chr(l_num + 0x60)
        decrypted += result + " "
    return decrypted.strip()


def day4(example=False, override=True, **kwargs):
    """Which room is it?"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "aaaaa-bbb-z-y-x-123[abxyz]\na-b-c-d-e-f-g-h-987[abcde]\nnot-a-real-room-404[oarel]\ntotally-real-room-200[decoy]"
    kiosk = get_input(day=day, seperator="\n", cast=_room_decode, override=override)
    sector_id_sum = 0
    for room in kiosk:
        name, freq_dict, sector_id, checksum = room
        freq_keys = sorted(freq_dict.keys(), reverse=True)
        my_checksum = ""
        for key in freq_keys:
            my_checksum += freq_dict[key]
        if my_checksum[:5] == checksum:
            sector_id_sum += int(sector_id)
            decrypted = _decrypt(name, int(sector_id))
            if "north" in decrypted:
                part_2_str = f"Part 2 The sector ID of the {decrypted} is {sector_id}"
    log.info(f"Part 1 sum of sector IDs is {sector_id_sum}")
    log.info(part_2_str)


def day5(example=False, override=None, **kwargs):  # noqa: ARG001
    """MD5 hash."""
    door_id = "abc" if example else "cxdnnyjw"
    passcode1 = ""
    passcode2: list[str | None] = [None] * 8
    i = 0
    while True:
        log.debug(i)
        if len(passcode1) == 8 and None not in passcode2:
            break
        h = hashlib.md5(f"{door_id}{i}".encode("utf-8")).hexdigest()  # noqa: UP012, S324
        if h.startswith("00000"):
            passcode1 += h[5] if len(passcode1) < 8 else ""
            position = int(h[5], 16)
            if position in range(8) and passcode2[position] is None:
                passcode2[position] = h[6]
            log.debug(passcode1)
            log.debug(passcode2)
        i += 1
    log.info(f"Part 1 passcode is {passcode1}")
    log.info(f"Part 2 passcode is {''.join(passcode2)}")  # type: ignore[arg-type]


def day6(example=False, override=False, **kwargs):
    """Please repeat!"""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("eedadn\ndrvtee\neandsr\nraavrd\natevrs\ntsrnev\nsdttsa\nrasrtv\n"
               "nssdts\nntnada\nsvetve\ntesnvt\nvntsnd\nvrdear\ndvrsen\nenarar\n")
    message = get_np_input(day=day, seperator="\n", cast=None, splitter=list, override=override)
    result1 = result2 = ""
    for column in message.T:
        values, counts = np.unique(column, return_counts=True)
        result1 += values[np.argmax(counts)]
        result2 += values[np.argmin(counts)]
    log.info(f"Part 1 the message was {result1}")
    log.info(f"Part 2 the message was {result2}")


def day7(example=False, override=False, **kwargs):
    """IPv7."""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("abba[mnop]qrst\n"
               "abcd[bddb]xyyx\n"
               "aaaa[qwer]tyui\n"
               "ioxxoj[asdfgh]zxcvbn\n")
    puzzle = get_input(day=day, seperator="\n", cast=None, override=override)
    inside_re = re.compile(r"\[(.*?)\]")
    outside_re = re.compile(r"([^[\]]+)(?:$|\[)")
    p1_count = 0
    for ip in puzzle:
        hypernet_abba = False
        for inside in inside_re.findall(ip):
            while len(inside) > 3 and not hypernet_abba:
                if (inside[0] != inside[1] and
                    inside[0] == inside[3] and
                    inside[1] == inside[2]):
                    hypernet_abba = True
                    break
                inside = inside[1:]  # noqa: PLW2901
            if hypernet_abba:
                break
        abba = False
        if not hypernet_abba:
            for outside in outside_re.findall(ip):
                while len(outside) > 3 and not abba:
                    if (outside[0] != outside[1] and
                        outside[0] == outside[3] and
                        outside[1] == outside[2]):
                        abba = True
                        break
                    outside = outside[1:]  # noqa: PLW2901
                if abba:
                    break
        if abba and not hypernet_abba:
            p1_count += 1
        log.debug("%s %s %s %d", ip, abba, hypernet_abba, p1_count)
    p2_count = 0
    for ip in puzzle:
        for outside in outside_re.findall(ip):
            ssl = False
            while len(outside) > 2:
                if (outside[0] != outside[1] and
                    outside[0] == outside[2]):
                    aba = f"{outside[1]}{outside[0]}{outside[1]}"
                    for inside in inside_re.findall(ip):
                        if aba in inside:
                            ssl = True
                            outside = ""  # noqa: PLW2901
                outside = outside[1:]  # noqa: PLW2901
            if ssl:
                break
        if ssl:
            p2_count += 1
    log.info(f"Part 1: {p1_count}")
    log.info(f"Part 2: {p2_count}")


def day8(example=False, override=False, **kwargs):
    """2FA."""
    _ = kwargs
    day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    if example:
        day = ("rect 3x2\n"
               "rotate column x=1 by 1\n"
               "rotate row y=0 by 4\n"
               "rotate column x=1 by 1\n")
    instructions = get_input(day=day, seperator="\n", cast=None, override=override)
    screen = np.full((6, 50), " ", dtype=str)
    for instruction in instructions:
        if "rect" in instruction:
            rows = int(instruction.split("x")[1])
            cols = int(instruction.split("x")[0].split()[1])
            screen[:rows, :cols] = "#"
        elif "column" in instruction:
            col = int(instruction.split("=")[1].split()[0])
            amount = int(instruction.split("by")[1].strip())
            screen[:, col] = np.roll(screen[:, col], amount)
        elif "row" in instruction:
            row = int(instruction.split("=")[1].split()[0])
            amount = int(instruction.split("by")[1].strip())
            screen[row] = np.roll(screen[row], amount)
        else:
            raise Exception(instruction)  # noqa: TRY002
    log.info(f"Part 1, {np.count_nonzero(screen == '#')} pixels are lit")
    log.info("Part 2 the code is:")
    print_np(screen)


def day9(part=1, example=False, override=False, **kwargs):
    """Decompression!"""
    # Ok, part 2 is kind of slow but it gets the correct answer so....
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = ("ADVENT\nA(1x5)BC\n(3x3)XYZ\nA(2x2)BCD(2x2)EFG\n(6x1)(1x3)A\nX(8x2)(3x3)ABCY\n"
               "(27x12)(20x12)(13x14)(7x10)(1x12)A\n(25x3)(3x3)ABC(2x3)XY(5x2)PQRSTX(18x9)(3x2)TWO(5x7)SEVEN")
    p = get_input(day=day, seperator="\n", cast=None, override=override)
    p1 = 0
    marker_re = re.compile(r"\([0-9]*x[0-9]*\)")
    for x in p:
        compressed = x
        t = 0
        while True:
            found = marker_re.search(compressed)
            if found:
                t += len(compressed[:found.start()])
                marker = compressed[found.start() + 1:found.end() - 1]
                chars, reps = map(int, marker.split("x"))
                decompressed = compressed[found.end():found.end() + chars] * reps
                if part == 1:
                    t += len(decompressed)
                    compressed = compressed[found.end() + chars:]
                else:
                    compressed = decompressed + compressed[found.end() + chars:]
            else:  # No more markers.
                t += len(compressed)
                break
        p1 += t
        log.debug("%s %d -> %d", x, t, p1)
    log.info(f"The decompressed length is : {p1}")


def day12(c=0, example=False, override=False, **kwargs):
    """More assembly code."""
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "cpy 41 a\ninc a\ninc a\ndec a\njnz a 2\ndec a"
    assembunny = get_input(day=day, seperator="\n", cast=None, override=override)
    ip = 0
    registers = {"a": 0, "b": 0, "c": int(c)}
    while ip < len(assembunny):
        op = assembunny[ip][:3]
        match op:
            case "cpy":
                value, reg = assembunny[ip][4:].split(" ")
                value = registers[value] if value.isalpha() else int(value)
                registers[reg] = int(value)
            case "inc":
                registers[assembunny[ip][4]] += 1
            case "dec":
                registers[assembunny[ip][4]] -= 1
            case "jnz":
                x, value = assembunny[ip][4:].split(" ")
                x = registers[x] if x.isalpha() else int(x)
                if x != 0:
                    ip += int(value)
                    continue
        ip += 1
    log.info(f"Register a is: {registers['a']}")


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
                           help="Set the debug logging level.")
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
