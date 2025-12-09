"""
Advent of Code.

Never did spend the time to work out how to get oAuth to work so this code expects you to
manually copy over your session cookie value.
Using a web browser inspect the cookies when logged into the Advent of Code website.
Copy the value from the "session" cookie into a text file called "session.txt"
"""  # noqa: CPY001

import collections
import hashlib
import heapq
import inspect
import itertools
import json
import logging
import pathlib
import pickle
import random
import re
import socket
import sys
from argparse import ArgumentParser
from concurrent import futures
from functools import reduce
from os import path

import numpy as np
import requests

# The Value from the session cookie used to make the webaccess.
# You could hardcode this with your value or set it at the interactive prompt.
# This is because I am lazy and didn't want to figure out how to scrape the cookie or work with the OAuth.
# I'd never work on these at the office but...

# Constants
_NAME = pathlib.Path(__file__).stem
_CODE_PATH = r"c:\AoC"
_YEAR = 2015
_OFFLINE = False

# Create and configure logger
logging.basicConfig(filename=f"{_NAME}.log", format="%(asctime)s %(message)s", filemode="a")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("").addHandler(console)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.i = log.info  # shortcut


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
    if _OFFLINE:
        with open(_CODE_PATH + r"\{}\day{}.txt".format(_YEAR, day)) as file_handler:  # noqa: UP032, FURB101, PTH123
            data_list = file_handler.read().split(seperator)
    elif isinstance(day, str):  # An example string
        data_list = day.split(seperator)
    else:
        if not path.exists(_CODE_PATH + "/session.txt"):  # noqa: PTH110
            raise Exception("Using the web browser get the session cookie value\nand put it as a string in {}".format(_CODE_PATH + r"\session.txt"))  # noqa: W605, EM103, TRY002
        with open(_CODE_PATH + "/session.txt", "r") as session_file:  # noqa: Q000, UP015, FURB101, PTH123
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
    if path.exists(_CODE_PATH + r"\{}\input.p".format(_YEAR)):  # noqa: UP032, PTH110, SIM108
        puzzle_dict = pickle.load(open(_CODE_PATH + r"\{}\input.p".format(_YEAR), "rb"))  # noqa: UP032, SIM115, PTH123, S301
    else:  # No pickle file, will need to make a new one.
        puzzle_dict = {}

    puzzle_input = puzzle_dict.get(day)

    if puzzle_input is None or override is True:
        puzzle_input = _pull_puzzle_input(day, seperator, cast)
        if isinstance(day, int):  # only save the full puzzle data to the pickle file.
            puzzle_dict[day] = puzzle_input
            pickle.dump(puzzle_dict, open(_CODE_PATH + r"\{}\input.p".format(_YEAR), "wb"))  # noqa: UP032, SIM115, PTH123, S301
    return puzzle_input


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


def day1(override=False, **kwargs):
    """So it begins."""
    _ = kwargs
    directions = get_input(day=1, seperator=None, cast=list, override=override)[0]
    log.debug(directions)
    log.info(f"Santa should go to floor {directions.count('(') - directions.count(')')}")
    for i in range(len(directions)):
        if directions[:i].count("(") - directions[:i].count(")") == -1:
            log.info(f"First enter basement at position {i}")
            break


def _day2_split(dimension_str):
    """Take the dimensions and return a sorted list."""  # noqa: DOC201
    return sorted(map(int, dimension_str.split("x")))


def day2(override=False, **kwargs):  # noqa: D103
    _ = kwargs
    packages = get_input(2, "\n", _day2_split, override)
    area = 0
    ribbon = 0
    for package in packages:
        area += 3 * package[0] * package[1] + 2 * package[1] * package[2] + 2 * package[2] * package[0]
        ribbon += (2 * sum(package[:2]) + np.prod(package))
    log.info(f"The elves need {area} sqft. of wrapping paper")
    log.info(f"The elves need {ribbon} ft. of ribbon")


def day3(example=False, override=False, **kwargs):  # noqa: D103
    _ = kwargs
    day: int | str = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))  # type: ignore[union-attr]
    if example:
        day = "^v^v^v^v^v"
    directions = get_input(day=day, seperator=None, override=override)[0]
    # log.info(directions)
    move_dict = {"^": Coordinate(( 0,  1)),  # noqa: E201, E241
                 ">": Coordinate(( 1,  0)),  # noqa: E201, E241
                 "<": Coordinate((-1,  0)),  # noqa: E241
                 "v": Coordinate(( 0, -1))}  # noqa: E201, E241
    santa_pos = Coordinate((0, 0))
    house_set = set()
    for move in directions:
        house_set.add(santa_pos)
        santa_pos += move_dict[move]
    log.info(f"{len(house_set)} houses received at least one package from Santa")
    santa_pos = Coordinate((0, 0))
    robot_pos = Coordinate((0, 0))
    house_set = set()
    counter = 0
    for move in directions:
        house_set.add(santa_pos)
        house_set.add(robot_pos)
        if counter % 2 == 0:
            santa_pos += move_dict[move]
        else:
            robot_pos += move_dict[move]
        counter += 1  # noqa: SIM113
    log.info(f"{len(house_set)} houses received at least one package from Santa or the Robot")


def _day4_thread(puzzle, check, this_range):  # noqa: D103
    for i in this_range:
        digest = hashlib.md5((puzzle + str(i)).encode("utf-8")).hexdigest()  # noqa: S324
        if digest[:len(check)] == check:
            log.info(f" Given {puzzle} and {check} the value could be {i}")
            return


def day4(check="00000", nmax=7, **kwargs):  # noqa: D103
    puzzle = "iwrupvqb"
    threads = 6
    with futures.ProcessPoolExecutor(max_workers=threads) as executor:
        running = []
        for thread in range(threads):
            r = range(1 + thread, 10**nmax + thread, threads)
            f = executor.submit(_day4_thread, puzzle, check, r)
            running.append(f)
        futures.wait(running, return_when=futures.ALL_COMPLETED)


def day5(**kwargs):  # noqa: D103
    _ = kwargs
    strings = get_input(5, "\n", None)
    nice = 0
    for s in strings:
        nice_check = 1
        if sum(s.count(x) for x in ["ab", "cd", "pq", "xy"]):
            nice_check = 0
        if nice_check and sum(s.count(x) for x in ["a", "e", "i", "o", "u"]) < 3:
            nice_check = 0
        if nice_check:
            nice_check = 0
            for i in range(len(s) - 1):
                if s[i] == s[i + 1]:
                    nice_check = 1
                    break
        nice += nice_check
    log.info(f"Part 1: {nice} strings are nice")
    nice = 0
    for s in strings:
        nice_check = 0
        for i in range(len(s) - 2):
            if s[i] == s[i + 2]:
                nice_check = 1
                break
        if nice_check:
            pair = ""
            nice_check = 0
            for i in range(len(s) - 1):
                pair = s[i:i + 2]
                # Replace the pair with some dummy characters.
                test_str = s[:i] + "00" + s[i + 2:]
                # Use regex to check for a match.
                if re.search(pair, test_str):
                    nice_check = 1
                    break
        # if nice_check:
        #    log.info("{} {}".format(pair,s))
        nice += nice_check
    log.info(f"Part 2: {nice} strings are nice")


def day6(**kwargs):  # noqa: D103
    _ = kwargs
    commands = get_input(6, "\n", None)
    # commands = ["turn on 0,0 through 0,0","toggle 0,0 through 9,9"]
    lights_on = set()
    brightness = {}
    for command_str in commands:
        if "toggle" in command_str:
            command, start, _, stop = command_str.split()
        else:
            _, command, start, _, stop = command_str.split()
        startx, starty = map(int, start.split(","))
        stopx, stopy = map(int, stop.split(","))
        for position in itertools.product(range(startx, stopx + 1), range(starty, stopy + 1)):
            brightness.setdefault(position, 0)
            if command == "on":
                lights_on.add(position)
                brightness[position] += 1
            if command == "off":
                lights_on.discard(position)
                brightness[position] = max(brightness[position] - 1, 0)
            if command == "toggle":
                brightness[position] += 2
                if position in lights_on:
                    lights_on.discard(position)
                else:
                    lights_on.add(position)
    log.info("Part1")
    log.info(f" {len(lights_on)} lights are on")
    total_brightness = 0
    total_brightness = sum(brightness.values())
    log.info("Part 2")
    log.info(f" Total brightness is {total_brightness}")


def day7(**kwargs):  # noqa: D103
    _ = kwargs
    connections = [
    "123 -> x",
    "456 -> y",
    "x AND y -> d",
    "x OR y -> e",
    "x LSHIFT 2 -> f",
    "y RSHIFT 2 -> g",
    "NOT x -> h",
    "NOT y -> i"]
    connections = get_input(7, "\n", None)
    wire_values = {}
    remaining_connections = list(connections)
    while len(remaining_connections) > 0:
        for connection in remaining_connections:
            # Parse the connection
            source2 = None
            connection_type = None
            connection_list = connection.split()
            if len(connection_list) == 3:
                source1, _, destination = connection_list
            elif len(connection_list) == 4:
                connection_type, source1, _, destination = connection_list
            elif len(connection_list) == 5:
                source1, connection_type, source2, _, destination = connection_list
                if source1.isnumeric():  # reversing the order here.
                    source2 = source1
                    source1 = connection_list[2]
            else:
                log.info("error")
                log.info(connection)
            # Make the connection (if possible)
            if source1.isnumeric():  # external signal input to wire.
                if destination == "b":
                    source1 = 46065
                wire_values[destination] = int(source1)
                remaining_connections.remove(connection)  # noqa: B909
            if source1 in wire_values:  # connection to an existing wire.
                if source2 is None:  # NOT or regular connection
                    if connection_type == "NOT":
                        wire_values[destination] = wire_values[source1] ^ 0xFFFF
                    else:  # regular connection
                        wire_values[destination] = wire_values[source1]
                    remaining_connections.remove(connection)  # noqa: B909
                elif source2 is not None:  # Joining two signals
                    if source2.isnumeric():  # external signal
                        source2 = int(source2)
                    else:  # Other wire
                        if source2 not in wire_values:
                            continue
                        source2 = wire_values[source2]
                    if connection_type == "RSHIFT":
                        wire_values[destination] = wire_values[source1] >> source2
                    elif connection_type == "LSHIFT":
                        wire_values[destination] = wire_values[source1] << source2
                    elif connection_type == "AND":
                        wire_values[destination] = wire_values[source1] & source2
                    elif connection_type == "OR":
                        wire_values[destination] = wire_values[source1] | source2
                    else:
                        raise Exception(f"Unsupported: {connection}")  # noqa: EM102, TRY002, TRY003
                    remaining_connections.remove(connection)  # noqa: B909
    # log.info("Part 1")
    log.info("Part 2")
    log.info(f" wire a is: {wire_values['a']}")


def day8(**kwargs):  # noqa: D103
    _ = kwargs
    # This should be a lot easier but I'm fighting with the Python string libraries
    santa = r'""\n"abc"\n"aaa\"aaa"\n"\x27"'  # Test input
    santa = _pull_puzzle_input(8, " ", None)[0]
    santa = santa.replace(r'"\n"', '""')
    code_len = sum(map(len, santa))
    log.info("Part 1")
    # Time to brute force count the characters
    string_values = 0
    index = 0
    while index < len(santa):
        if santa[index] != '"':
            # log.debug(santa[index])
            if santa[index] == "\\":
                if santa[index + 1] in {'"', "\\"}:
                    index += 1
                elif santa[index + 1] == "x":
                    index += 3
                else:
                    log.info(santa[index - 5:index + 5])
                    raise Exception(f"index: {index}")  # noqa: EM102, TRY002, TRY003
            string_values += 1
        index += 1
    log.info(f" code length: {code_len}")
    log.info(f" string values: {string_values}")
    log.info(f" answer: {code_len - string_values}")
    log.info("Part 2")
    # Just used VIM
    open_quotes = 300 * 6  # Open close quotes turns into 6 char
    backslash = 645 * 2
    single_quotes = 229 * 2
    characters = 4728
    string_values = open_quotes + backslash + single_quotes + characters
    log.info(f" string values: {string_values}")
    log.info(f" code length: {code_len}")
    log.info(f" answer: {string_values - 6202}")


def day9(**kwargs):  # noqa: D103
    _ = kwargs
    distances = get_input(9, "\n", lambda a: (a.split(" ")[0], a.split(" ")[2], int(a.split(" ")[4])))
    # distances = [("London","Dublin",464),("London","Belfast",518),("Dublin","Belfast",141)]
    city_set = set()
    distance_dict = {}
    for city_a, city_b, distance in distances:
        city_set.add(city_a)
        city_set.add(city_b)
        distance_dict.setdefault(city_a, {})
        distance_dict[city_a][city_b] = distance
        distance_dict.setdefault(city_b, {})
        distance_dict[city_b][city_a] = distance
    min_distance = 2**24
    max_distance = 0
    for route in itertools.permutations(city_set):
        route_distance = 0
        for index in range(len(route)):
            if index + 1 == len(route):
                continue
            route_distance += distance_dict[route[index]][route[index + 1]]
        min_distance = min(min_distance, route_distance)
        max_distance = max(max_distance, route_distance)
        log.debug(f"route={route} distance={route_distance}")
    log.info(f"Part 1 shortest distance is {min_distance}")
    log.info(f"Part 2 longest distance is {max_distance}")


def day10(**kwargs):  # noqa: D103
    _ = kwargs
    """Look, say."""
    puzzle = "3113322113"
    # puzzle = "1"
    output = ""
    for process in range(50):
        output = ""
        test_char = puzzle[0]
        counter = 1
        for index in range(1, len(puzzle)):
            if puzzle[index] != test_char:
                output += str(counter)
                output += test_char
                test_char = puzzle[index]
                counter = 1
            else:
                counter += 1
        output += str(counter)
        output += test_char
        log.debug(f"process={process}, test_char={test_char}, puzzle={puzzle}, output={output}")
        puzzle = output
        if process == 39:
            log.info(f"Part 1 length = {len(output)}")
    log.info(f"Part 2 length = {len(output)}")


def _test_password(password):
    debug_print = False
    if any(x in password for x in ["i", "o", "l"]):
        if debug_print:
            log.info(f"{password} contains i,o or l")
        return False
    first_pair = None
    found_match = False
    found_inc = False
    for index in range(len(password) - 2):
        if ord(password[index + 2]) - ord(password[index + 1]) == 1\
           and ord(password[index + 1]) - ord(password[index]) == 1:
            found_inc = True
        if ord(password[index + 2]) - ord(password[index + 1]) == 0:
            if first_pair is not None and first_pair != password[index + 1]:
                found_match = True
            else:
                first_pair = password[index + 1]
            if debug_print:
                log.info(f"index={index} pair {password[index + 2]}{password[index + 1]} {found_match}, {first_pair}")
        if ord(password[index + 1]) - ord(password[index]) == 0:
            if first_pair is not None and first_pair != password[index]:
                found_match = True
            else:
                first_pair = password[index]
            if debug_print:
                log.info(f"index={index} pair {password[index + 1]}{password[index]} {found_match}, {first_pair}")
    if debug_print:
        if found_match is False:
            log.info(f"{password} does not contain two matching pairs")
        if found_inc is False:
            log.info(f"{password} does not contain incrementing sequence")
    return found_match and found_inc


def _increment(password):
    password_list = list(map(ord, list(password)))
    password_list.reverse()
    password_list[0] += 1
    while any(x > 0x7a for x in password_list):
        for index in range(len(password_list)):
            if password_list[index] > 0x7a:
                password_list[index] = 0x61
                if index < len(password_list) - 1:
                    password_list[index + 1] += 1
    password_list.reverse()
    output_str = ""
    for char in password_list:
        output_str += chr(char)
    return output_str


def day11(**kwargs):
    """Corporate Policy."""
    _ = kwargs
    password = "hepxcrrq"  # noqa: S105
    while _test_password(password) is False:
        password = _increment(password)
    log.info(f"Part 1 new password is {password}")
    password = _increment(password)
    while _test_password(password) is False:
        password = _increment(password)
    log.info(f"Part 2 new password is {password}")


def _recursive_sum(thing, current_sum=0, part2=False):
    if type(thing) is list:
        for item in thing:
            if type(item) is int:
                current_sum += item
            if type(item) in {list, dict}:
                current_sum = _recursive_sum(item, current_sum, part2)
    if type(thing) is dict:
        if part2 and ("red" in thing or "red" in thing.values()):
            return current_sum
        for key, value in thing.items():
            if type(key) is int:
                current_sum += key
            if type(value) is int:
                current_sum += value
            if type(value) in {list, dict}:
                current_sum = _recursive_sum(value, current_sum, part2)
    return current_sum


def day12(override=False, **kwargs):  # noqa: D103
    """JSON accounting!"""
    _ = kwargs
    data = json.loads(get_input(day=12, seperator=None, cast=None, override=override)[0])
    log.info(f"Part 1 sum is {_recursive_sum(data)}")
    log.info(f"Part 2 sum is {_recursive_sum(data, part2=True)}")


def day13(part="1", **kwargs):  # noqa: D103
    _ = kwargs
    happiness = [
        ["Alice", "would", "gain", "54", "happiness", "units", "by", "sitting", "next", "to", "Bob."  ],  # noqa: E202, W291
        ["Alice", "would", "lose", "79", "happiness", "units", "by", "sitting", "next", "to", "Carol."],
        ["Alice", "would", "lose", "2",  "happiness", "units", "by", "sitting", "next", "to", "David."],  # noqa: E241
        ["Bob",   "would", "gain", "83", "happiness", "units", "by", "sitting", "next", "to", "Alice."],  # noqa: E241
        ["Bob",   "would", "lose", "7",  "happiness", "units", "by", "sitting", "next", "to", "Carol."],  # noqa: E241
        ["Bob",   "would", "lose", "63", "happiness", "units", "by", "sitting", "next", "to", "David."],  # noqa: E241
        ["Carol", "would", "lose", "62", "happiness", "units", "by", "sitting", "next", "to", "Alice."],
        ["Carol", "would", "gain", "60", "happiness", "units", "by", "sitting", "next", "to", "Bob."  ],  # noqa: E202, W291
        ["Carol", "would", "gain", "55", "happiness", "units", "by", "sitting", "next", "to", "David."],
        ["David", "would", "gain", "46", "happiness", "units", "by", "sitting", "next", "to", "Alice."],
        ["David", "would", "lose", "7",  "happiness", "units", "by", "sitting", "next", "to", "Bob."  ],  # noqa: E202, E241, W291
        ["David", "would", "gain", "41", "happiness", "units", "by", "sitting", "next", "to", "Carol."]]
    happiness = get_input(13, "\n", lambda a: a.split(" "))
    decoder = {"gain": 1, "lose": -1}
    happiness_dict = {"Me": {}}
    attendence_set = set()
    max_happiness = 0
    for happy_list in happiness:
        attendence_set.add(happy_list[0])
        happiness_dict.setdefault(happy_list[0], {})
        happiness_dict[happy_list[0]][happy_list[-1].rstrip(".")] = decoder[happy_list[2]] * int(happy_list[3])
        happiness_dict[happy_list[0]]["Me"] = 0
        happiness_dict["Me"][happy_list[0]] = 0
    if part == "2":
        attendence_set.add("Me")  # For Part 2 add me into the seating charts
    for placement in itertools.permutations(attendence_set):
        this_happiness = 0
        placement = list(placement)  # noqa: PLW2901
        for index in range(-1, len(placement) - 1):
            log.debug(f"  {placement[index]} to {placement[index + 1]}")
            this_happiness += happiness_dict[placement[index]][placement[index + 1]]
            this_happiness += happiness_dict[placement[index + 1]][placement[index]]
        log.debug(f"Placement {placement} = {this_happiness}")
        max_happiness = max(max_happiness, this_happiness)
    if "Me" not in attendence_set:
        log.info(f"Part 1 optimal happiness is {max_happiness}")
    else:
        log.info(f"Part 2 optimal happiness is {max_happiness}")


def day14(**kwargs):
    """Reindeer distance."""
    _ = kwargs
    speeds = [["Comet", "can", "fly", "14", "km/s", "for", "10", "seconds, ", "but", "then", "must", "rest", "for", "127", "seconds."],
              ["Dancer", "can", "fly", "16", "km/s", "for", "11", "seconds, ", "but", "then", "must", "rest", "for", "162", "seconds."]]
    time = 1000
    speeds = get_input(14, "\n", lambda a: a.split(" "))
    time = 2503
    reindeer_dict = {}
    for speed in speeds:
        reindeer_dict[speed[0]] = {"speed": int(speed[3]), "fly": int(speed[6]), "rest": int(speed[-2])}
        reindeer_dict[speed[0]]["total"] = reindeer_dict[speed[0]]["fly"] + reindeer_dict[speed[0]]["rest"]
        reindeer_dict[speed[0]]["position"] = 0
        reindeer_dict[speed[0]]["points"] = 0

    winner = [None, 0]
    log.debug(reindeer_dict)
    for deer, data in reindeer_dict.items():
        # Whole time interval distance travelled
        distance = (time // data["total"]) * data["fly"] * data["speed"]
        remaining_time = time - (data["total"] * (time // data["total"]))
        string = f" {deer} travelled {distance} in {time - remaining_time}s"
        # Remaining fractional interval calculation
        distance += data["fly"] * data["speed"] if remaining_time >= data["fly"] else remaining_time * data["speed"]
        log.info(f"{string} then to {distance} in the remaining {remaining_time}s")
        if distance > winner[1]:
            winner = [deer, distance]
    log.info(f"Part 1 {winner[0]} won with {winner[1]} distance")
    max_distance = 0
    for sec in range(1, time + 1):
        for data in reindeer_dict.values():
            interval = sec % data["total"]
            if interval != 0 and interval <= data["fly"]:
                data["position"] += data["speed"]
        # Track who is in front.
        deer_in_front = []
        for deer, data in reindeer_dict.items():
            if data["position"] > max_distance:
                max_distance = data["position"]
                deer_in_front = [deer]
            elif data["position"] == max_distance:
                deer_in_front.append(deer)
        # Award the point(s)
        for deer in deer_in_front:
            reindeer_dict[deer]["points"] += 1
        log.debug(f"After {sec} seconds")
        for data in reindeer_dict.values():
            log.debug(f" {deer} at {data['position']} with {data['points']} points")
        # foo = input()
    winner = [None, 0]
    for deer, data in reindeer_dict.items():
        log.info(f" {deer} has {data['points']} points")
        if data["points"] > winner[1]:
            winner = [deer, data["points"]]
    log.info(f"Part 2 {winner[0]} won with {winner[1]} points")


def day15(part2=False, **kwargs):  # noqa: D103
    _ = kwargs
    day = "Butterscotch: capacity -1, durability -2, flavor 6, texture 3, calories 8\nCinnamon: capacity 2, durability 3, flavor -2, texture -1, calories 3"
    day = 15
    ingredients = get_input(day, "\n", None)
    ingredients_dict = {}
    properties_set = set()
    for item in ingredients:
        name, values = item.split(":")
        value_dict = {}
        value_list = values.strip().split(",")
        for value in value_list:
            properties, num = value.strip().split(" ")
            value_dict[properties] = int(num)
            properties_set.add(properties)
        ingredients_dict[name] = value_dict
    ingredients_list = list(ingredients_dict.keys())
    max_score = 0
    tsp_range = range(101)
    log.debug(ingredients_dict)
    for amounts in itertools.product(*[tsp_range] * len(ingredients_dict.keys())):
        if sum(amounts) != 100:
            continue
        log.debug(amounts)
        score = 1
        cal = 0
        for i in range(len(amounts)):
            cal += (ingredients_dict[ingredients_list[i]]["calories"] * amounts[i])
        if cal != 500 and part2:
            continue
        for p in properties_set:
            cal = 0
            if p == "calories":
                continue
            p_score = 0
            for i in range(len(amounts)):
                p_score += (ingredients_dict[ingredients_list[i]][p] * amounts[i])
            p_score = max(0, p_score)
            if p_score == 0:
                score = 0
                break
            score *= p_score
        max_score = max(max_score, score)
    log.info(f"Part {2 if part2 else 1}: {max_score}")


_position_dict = {"children": 0 * 8,
                 "cats": 1 * 8,
                 "samoyeds": 2 * 8,
                 "pomeranians": 3 * 8,
                 "akitas": 4 * 8,
                 "vizslas": 5 * 8,
                 "goldfish": 6 * 8,
                 "trees": 7 * 8,
                 "cars": 8 * 8,
                 "perfumes": 9 * 8}


def _mfcsam_encode(string):
    """
    Create a match/mask set for the mfcsam data.

    Assumes that item encoding/quantity is < 256

    Returns:
        (int, int)
    """
    match = 0
    mask = 0
    item_list = string.split(", ")
    for item in item_list:
        key, value = item.split(": ")
        match |= int(value) << _position_dict[key]
        mask |= 0xFF << _position_dict[key]
    return match, mask


def day16(**kwargs):  # noqa: D103
    _ = kwargs
    mfcsam_data = "children: 3, cats: 7, samoyeds: 2, pomeranians: 3, akitas: 0, vizslas: 0, goldfish: 5, trees: 3, cars: 2, perfumes: 1"
    sue_data_list = get_input(16, "\n", None)
    match, _ = _mfcsam_encode(mfcsam_data)
    for sue_data in sue_data_list:
        sue = sue_data.split(": ")[0]
        knowledge = sue_data.replace(sue + ": ", "")
        sue_match, mask = _mfcsam_encode(knowledge)
        if match & mask == sue_match:
            log.info(f"Part 1 {sue} is a match")
            break
    # Part 2, just going to hack in a fix.
    greater = ["cats", "trees"]
    less = ["pomeranians", "goldfish"]
    adjustment_list = greater + less
    adjustment_mask = 0
    for adjustment in adjustment_list:
        adjustment_mask |= 0xFF << _position_dict[adjustment]
    log.debug(f"adjustment mask = {adjustment_mask:#x}")
    for sue_data in sue_data_list:
        sue = sue_data.split(": ")[0]
        knowledge = sue_data.replace(sue + ": ", "")
        sue_match, mask = _mfcsam_encode(knowledge)
        # First check the exact values
        if match & (mask & ~adjustment_mask) == sue_match:
            log.debug(f"{sue} - {sue_match:#022x}, {mask:#022x}, {mask & ~adjustment_mask:#022x}")
            # Next check the special case for the things we know
            match = True
            for thing in greater:
                log.debug(f"{thing} = {(mask >> _position_dict[thing]) & 0xFF}")
                if (mask >> _position_dict[thing]) & 0xFF == 0xFF and (sue_match >> _position_dict[thing]) & 0xFF < (match >> _position_dict[thing]):
                    match = False
            for thing in less:
                log.debug(f"{thing} = {(mask >> _position_dict[thing]) & 0xFF} and {(sue_match >> _position_dict[thing]) & 0xFF}"
                          f">= {(match >> _position_dict[thing])}")
                if (mask >> _position_dict[thing]) & 0xFF == 0xFF and (sue_match >> _position_dict[thing]) & 0xFF > (match >> _position_dict[thing]):
                    match = False
            if match is True:
                log.info(f"Part 2 {sue} is a match")
                break


def _find_combinations(containers, final, container_count):
    return [pair for pair in itertools.combinations(containers, container_count) if sum(pair) == final]


def day17(**kwargs):
    """Eggnog."""
    _ = kwargs
    containers = [20, 15, 10, 5, 5]
    total = 25
    containers = get_input(17, "\n", int)
    total = 150
    possible_combinations = 0
    for i in range(len(containers) + 1):
        possible_combinations += len(_find_combinations(containers, total, i))
    log.info(f"Part 1 number of possible combinations is {possible_combinations}")
    for i in range(len(containers) + 1):
        possible_combinations = _find_combinations(containers, total, i)
        if len(possible_combinations) != 0:
            log.info(f"Part 2 minimum is {i} containers there are {len(possible_combinations)} combinations")
            break


def day18(**kwargs):
    """Santa's lights (game of life)."""
    _ = kwargs
    initial_state = [".#.#.#",
                     "...##.",
                     "#....#",
                     "..#...",
                     "#.#..#",
                     "####.."]
    steps = 4
    initial_state = ["##.#.#",
                     "...##.",
                     "#....#",
                     "..#...",
                     "#.#..#",
                     "####.#"]
    steps = 5
    initial_state = get_input(18, "\n", None)
    steps = 100
    initial_list = []
    for s in initial_state:
        l = []  # noqa: E741
        for c in s:
            if c == ".":
                l.append(0)
            else:
                l.append(1)
        initial_list.append(l)
    lights = np.array(initial_list)
    updates = np.copy(lights)
    log.debug(lights)
    for _ in range(steps):
        for row in range(lights.shape[0]):
            for col in range(lights.shape[1]):
                test_array = lights[max(row - 1, 0):min(row + 2, lights.shape[0]), max(col - 1, 0): min(col + 2, lights.shape[1])]
                if lights[row, col] == 1 and np.sum(test_array) not in {3, 4}:  # Light was on, turns off
                    updates[row, col] = 0
                elif lights[row, col] == 0 and np.sum(test_array) == 3:
                    updates[row, col] = 1
                else:
                    updates[row, col] = lights[row, col]
        lights = np.copy(updates)
    log.info(f"Part 1: After {steps} steps there were {np.sum(lights)} lights lit")
    lights = np.array(initial_list)
    updates = np.copy(lights)
    log.debug(lights)
    for _ in range(steps):
        # Stuck lights
        lights[0, 0] = 1
        lights[0, lights.shape[1] - 1] = 1
        lights[lights.shape[0] - 1, 0] = 1
        lights[lights.shape[0] - 1, lights.shape[1] - 1] = 1
        for row in range(lights.shape[0]):
            for col in range(lights.shape[1]):
                test_array = lights[max(row - 1, 0):min(row + 2, lights.shape[0]), max(col - 1, 0): min(col + 2, lights.shape[1])]
                if lights[row, col] == 1 and np.sum(test_array) not in {3, 4}:  # Light was on, turns off
                    updates[row, col] = 0
                elif lights[row, col] == 0 and np.sum(test_array) == 3:
                    updates[row, col] = 1
                else:
                    updates[row, col] = lights[row, col]
        lights = np.copy(updates)
        log.debug(lights)
        # _ = input()
    # Stuck lights
    lights[0, 0] = 1
    lights[0, lights.shape[1] - 1] = 1
    lights[lights.shape[0] - 1, 0] = 1
    lights[lights.shape[0] - 1, lights.shape[1] - 1] = 1
    log.info(f"Part 2: After {steps} steps there were {np.sum(lights)} lights lit")


def day19(**kwargs):
    """HOHOHO."""
    _ = kwargs
    # Parsing the data
    puzzle = ["H => HO",
              "H => OH",
              "O => HH",
              "",
              "HOHOHO"]
    # rules = {} can't use a dictionary there are duplicate keys.
    puzzle = get_input(19, "\n", None)
    machine_in = []
    machine_out = []
    for line in puzzle:
        if not line:
            break
        i, o = line.split(" => ")
        machine_in.append(i)
        machine_out.append(o)
    machine_zip = zip(machine_in, machine_out, strict=True)
    molecule = puzzle[puzzle.index("") + 1]

    # Solve part 1
    molecule_set = set()
    for machine_input, machine_output in machine_zip:
        log.debug(f"Testing {machine_input} => {machine_output}")
        for i in range(len(molecule)):
            if i + len(machine_input) > len(molecule):
                continue
            log.debug(f" i={i} checking {molecule[i:i + len(machine_input)]}")
            if molecule[i:i + len(machine_input)] == machine_input:
                new = molecule[:i] + machine_output + molecule[i + len(machine_input):]
                log.debug(f" new molecule = {new}")
                molecule_set.add(new)
    log.info(f"Part 1 there are {len(molecule_set)} distinct molecules")


def day19_p2(example=False, **kwargs):
    """Randomly try to reverse the molecule..."""
    _ = kwargs
    day = 19 if not example else ("e => H\n"
                                  "e => O\n"
                                  "H => HO\n"
                                  "H => OH\n"
                                  "O => HH\n"
                                  "\n"
                                  "HOHOHO\n")
    replacements = get_input(day, "\n", None)
    molecule = replacements[-1]
    replacements = replacements[:-1]
    transforms = []
    for replacement in replacements:
        if "=>" in replacement:
            frm, to = replacement.split(" => ")
            transforms.append((frm, to))
    count = tries = 0
    working_mol = molecule
    while len(working_mol) > 1:
        start = working_mol
        for frm, to in transforms:
            while to in working_mol:
                count += working_mol.count(to)
                working_mol = working_mol.replace(to, frm)
        if start == working_mol:  # no progress
            random.shuffle(transforms)
            working_mol = molecule
            count = 0
            tries += 1
    log.info(f"{count} transformation after {tries} tries")


def _visiting_elves(house):
    visiting_elves = []
    test_value = house
    while test_value > 0:
        if house % test_value == 0:
            visiting_elves.append(test_value)
        test_value -= 1
    return visiting_elves


def _presents(house):
    test_value = house
    presents = 0
    while test_value > 0:
        if house % test_value == 0:
            presents += test_value
        test_value -= 1
    return presents * 10


def _presents_part2(house):
    test_value = house
    presents = 0
    while test_value > 0:
        if house % test_value == 0 and test_value * 50 >= house:
            presents += test_value
        test_value -= 1
    return presents * 11


def day20(**kwargs):  # noqa: D103
    _ = kwargs
    lots_of_houses = 1000000
    goal = 29000000
    # np arrays of lots of houses.
    part1_houses = np.zeros(lots_of_houses)
    part2_houses = np.zeros(lots_of_houses)
    for elf in range(1, lots_of_houses):
        # Walk array starting at elf and incrementing by elf, each time deliver elf * 10 packages.
        part1_houses[elf::elf] += (10 * elf)
        # Walk the array starting at elf and stopping after 50 houses (+1 for the slice), each time deliver 11 * elf packages.
        part2_houses[elf:(elf + 1) * 50:elf] += 11 * elf

    # Use np nonzero to find all the houses that have >= goal packages then grab the first one (lowest) off of the list
    log.info(f"There were {np.nonzero(part1_houses >= goal)[0].size} houses that were >= {goal}")
    log.info(f"Part 1 - {np.nonzero(part1_houses >= goal)[0][0]}")
    log.info(f"Part 2 - {np.nonzero(part2_houses >= goal)[0][0]}")


def day21(**kwargs):
    """RPG Sim."""
    _ = kwargs
    weapons = {
        "Dagger":     {"cost": 8, "damage": 4},
        "Shortsword": {"cost": 10, "damage": 5},
        "Warhammer":  {"cost": 25, "damage": 6},
        "Longsword":  {"cost": 40, "damage": 7},
        "Greataxe":   {"cost": 74, "damage": 8}}
    armors = {
        "None":       {"cost": 0, "armor": 0},
        "Leather":    {"cost": 13, "armor": 1},
        "Chainmail":  {"cost": 31, "armor": 2},
        "Splintmail": {"cost": 53, "armor": 3},
        "Bandedmail": {"cost": 75, "armor": 4},
        "Platemail":  {"cost": 102, "armor": 5}}
    rings = {
        "None":       {"cost": 0},
        "Damage +1":  {"cost": 25, "damage":  1},
        "Damage +2":  {"cost": 50, "damage":  2},
        "Damage +3":  {"cost": 100, "damage": 3},
        "Defense +1": {"cost": 20, "armor": 1},
        "Defense +2": {"cost": 40, "armor": 2},
        "Defense +3": {"cost": 80, "armor": 3}}
    # Lazy loop search
    enemy = {"hp": 109, "damage": 8, "armor": 2}
    cost_of_winning_fights = []
    cost_of_losing_fights = []
    for weapon in weapons.values():
        for armor in armors.values():
            for r1_name, ring1 in rings.items():
                for r2_name, ring2 in rings.items():
                    if r1_name == r2_name and r1_name != "None":
                        continue  # Can't buy two of the same ring.
                    # log.info(f"Fighting with {w_name}, {a_name}, {r1_name} & {r2_name}")
                    damage = weapon["damage"] + ring1.get("damage", 0) + ring2.get("damage", 0)
                    defense = armor["armor"] + ring1.get("armor", 0) + ring2.get("armor", 0)
                    cost = weapon["cost"] + armor["cost"] + ring1.get("cost", 0) + ring2.get("cost", 0)
                    player_turns = enemy["hp"] if damage - enemy["armor"] <= 0 else enemy["hp"] / (damage - enemy["armor"])
                    enemy_turns = 100 if enemy["damage"] - defense <= 0 else 100 / (enemy["damage"] - defense)
                    if player_turns <= enemy_turns:
                        # log.info("Win!")
                        cost_of_winning_fights.append(cost)
                    else:
                        cost_of_losing_fights.append(cost)
    log.info(f"Part 1 - The cheapest winning fight was {min(cost_of_winning_fights)}")
    log.info(f"Part 2 - The most expensive losing fight was {max(cost_of_losing_fights)}")


class Spell(collections.namedtuple("base_spell", "name cost effect turns damage heal armor mana")):  # noqa: D101, PYI024, SLOT002
    def __new__(cls, name, cost, effect=False, turns=None, damage=0, heal=0, armor=0, mana=0):  # noqa: D102
        return super().__new__(cls, name, cost, effect, turns, damage, heal, armor, mana)


spells = (
    Spell("Magic Missile", 53,  damage=4),  # noqa: E241
    Spell("Drain",         73,  damage=2, heal=2),  # noqa: E241
    Spell("Shield",        113, effect=True, turns=6, armor=7),  # noqa: E241
    Spell("Poison",        173, effect=True, turns=6, damage=3),  # noqa: E241
    Spell("Recharge",      229, effect=True, turns=5, mana=101),  # noqa: E241
)


class State(object):  # noqa: D101, UP004
    def __init__(self, hp, mana, boss_hp, boss_damage, mana_spent=0, effects=None, hard=False, parent=None, spell_cast=None):  # noqa: D107
        self.hp = hp
        self.mana = mana
        self.boss_hp = boss_hp
        self.boss_damage = boss_damage
        self.mana_spent = mana_spent
        self.effects = effects or ()
        self.hard = hard
        self._parent = parent
        self._spell_cast = spell_cast

    def __eq__(self, other):  # noqa: D105
        if not isinstance(other, State):
            return NotImplemented
        return all(getattr(self, k) == getattr(other, k) for k in vars(self) if k[0] != "_")

    def __hash__(self):  # noqa: D105
        return reduce(lambda a, b: a ^ hash(b), (v for k, v in vars(self).items() if k[0] != "_"), 0)

    def iter_path(self):  # noqa: D102
        if self._parent is None:
            return
        yield from self._parent.iter_path()
        yield self._spell_cast

    def process_effects(self, hp, mana, boss_hp):  # noqa: D102
        remaining_effects = []
        armor = 0
        for timer, effect in self.effects:
            hp += effect.heal
            mana += effect.mana
            boss_hp -= effect.damage
            armor = max(armor, effect.armor)
            if timer > 1:
                remaining_effects.append((timer - 1, effect))
        return tuple(remaining_effects), hp, mana, boss_hp, armor

    def boss_turn(self):  # noqa: D102
        self.effects, self.hp, self.mana, self.boss_hp, armor = (self.process_effects(self.hp, self.mana, self.boss_hp))
        if self.boss_hp > 0:
            self.hp -= max(1, self.boss_damage - armor)

    def transitions(self):  # noqa: D102
        # Plater first
        effects, hp, mana, boss_hp, __ = self.process_effects(self.hp - int(self.hard), self.mana, self.boss_hp)
        for spell in spells:
            if spell.cost > mana or any(spell is s for t, s in effects):
                continue  # Skip if not enough mana or in effect.
            new_state = State(hp, mana - spell.cost, boss_hp, self.boss_damage, self.mana_spent + spell.cost,
                              effects, hard=self.hard, parent=self, spell_cast=spell.name)
            if not spell.effect:
                new_state.hp += spell.heal
                new_state.boss_hp -= spell.damage
            else:
                new_state.effects += ((spell.turns, spell),)
            # Boss turn
            new_state.boss_turn()
            # Stop if dead.
            if new_state.hp > 0:
                yield new_state


def search_a_star(start):  # noqa: D103
    open_states = {start}
    pqueue = [(0, start)]
    closed_states = set()
    unique = itertools.count()
    while open_states:
        current = heapq.heappop(pqueue)[-1]
        if current.boss_hp < 1:
            return current
        open_states.remove(current)
        closed_states.add(current)
        for state in current.transitions():
            if state in closed_states or state in open_states:
                continue
            open_states.add(state)
            heapq.heappush(pqueue, (state.mana_spent, next(unique), state))
    return current


def day22(**kwargs):  # noqa: D103
    _ = kwargs
    boss_hp = 55
    boss_damage = 8
    player_hp = 50
    player_mana = 500
    start = State(player_hp, player_mana, boss_hp, boss_damage)
    end = search_a_star(start)
    log.info(f"Part 1: {end.mana_spent}")

    start.hard = True
    end = search_a_star(start)
    log.info(f"Part 2: {end.mana_spent}")


class Computer:  # noqa: D101
    def __init__(self, **kwargs):  # noqa: D107
        self.registers = {"a": 0, "b": 0}
        self.instructions = {}
        self.ip = 0
        self.load(kwargs.get("puzzle", []))

    def load(self, puzzle):  # noqa: D102
        for ip, line in enumerate(puzzle):
            opcode = line.split(" ")[0]
            variable = line.replace(f"{opcode} ", "")
            self.instructions[ip] = [opcode, variable]

    def run(self):  # noqa: D102
        while self.instructions.get(self.ip, False):
            opcode, variable = self.instructions[self.ip]
            log.debug(f"IP={self.ip} OP={opcode} {variable}")
            if opcode == "hlf":
                self.registers[variable] //= 2
            elif opcode == "tpl":
                self.registers[variable] *= 3
            elif opcode == "inc":
                self.registers[variable] += 1
            elif opcode == "jmp":
                self.ip += int(variable)
                continue
            elif opcode == "jie":
                register, offset = variable.split(", ")
                if self.registers[register] % 2 == 0:
                    self.ip += int(offset)
                    continue
            elif opcode == "jio":
                register, offset = variable.split(", ")
                if self.registers[register] == 1:
                    self.ip += int(offset)
                    continue
            else:
                raise Exception(f"Unknown opcode {opcode} at {self.ip}")  # noqa: W605, EM102, EM103, TRY002, TRY003
            log.debug(self.registers)
            self.ip += 1


def day23(**kwargs):  # noqa: D103
    _ = kwargs
    puzzle = ["inc a", "jio a, +2", "tpl a", "inc a"]
    puzzle = get_input(23, "\n", None)
    computer = Computer(puzzle=puzzle)
    computer.run()
    log.info("Part 1")
    for register, value in computer.registers.items():
        log.info(f" Register {register} is {value}")
    computer = Computer(puzzle=puzzle)
    computer.registers["a"] = 1
    computer.run()
    log.info("Part 2")
    for register, value in computer.registers.items():
        log.info(f" Register {register} is {value}")


def _knapsack(t, wt, val, n):
    knapsack = [[0 for x in range(t + 1)] for x in range(n + 1)]
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(t + 1):
            if i == 0 or w == 0:
                knapsack[i][w] = 0
            elif wt[i - 1] <= w:
                knapsack[i][w] = max(val[i - 1] + knapsack[i - 1][w - wt[i - 1]], knapsack[i - 1][w])
            else:
                knapsack[i][w] = knapsack[i - 1][w]

    return knapsack[n][t]


def day24(**kwargs):  # noqa: D103
    _ = kwargs
    # Driver code
    val = [60, 100, 120]
    wt = [10, 20, 30]
    t = 20
    n = 2
    log.info(_knapsack(t, wt, val, n))


def day25(**kwargs):  # noqa: D103
    _ = kwargs
    y = 1
    x = 1
    cur_val = 20151125
    mul = 252533
    div = 33554393
    while True:
        if y == 1:
            y = x + 1
            x = 1
        else:
            y -= 1
            x += 1
        cur_val = (cur_val * mul) % div
        if y % 100 == 0 and x == 1:
            log.info("%s %s %s", y, x, cur_val)
        if y == 2947 and x == 3029:
            log.info("%s %s %s", y, x, cur_val)
            break


def main(argv=None):
    """Command line interface."""
    argparser = ArgumentParser(prog=_NAME)
    argparser.add_argument("--day", type=int, default=1, help="AoC day to run.")
    argparser.add_argument("--example", dest="example", action="store_true", default=False,
                           help="Run the code against the puzzle example.")
    argparser.add_argument("--override", dest="override", action="store_true", default=False,
                           help="Override the stored puzzle input data.")
    argparser.add_argument("--debug", dest="debug", action="store_true", default=False,
                           help="Debug logging.")
    args, unknown = argparser.parse_known_args(argv)
    kwargs = {x.removeprefix("--").split("=")[0]: x.split("=")[1] for x in unknown}
    if args.debug:
        log.info("Setting debug logging level!")
        log.setLevel(logging.DEBUG)
    if hasattr(sys.modules[__name__], f"day{args.day}"):
        day = getattr(sys.modules[__name__], f"day{args.day}")
        day(**vars(args), **kwargs)
    else:
        log.info("%s.py does not have a function called day%s", _NAME, args.day)


if __name__ == "__main__":
    sys.exit(main())
