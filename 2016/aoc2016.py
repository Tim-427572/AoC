import socket
import requests
import pickle
import numpy as np
from os import path
import statistics
import math
import time
import re
import copy
import sys
from collections import defaultdict
import itertools
from functools import lru_cache
from string import ascii_lowercase

# Advent of Code
# Never did spend the time to work out how to get oAuth to work so this code expects you to
# manually copy over your session cookie value.
# Using a web browser inspect the cookies when logged into the Advent of Code website.
# Copy the value from the "session" cookie into a text file called "session.txt"

# Constants
_code_path = r'c:\AoC'
_offline = False
_year = 2016


def _check_internet(host="8.8.8.8", port=53, timeout=2):
    """
    Attempt to check for the firewall by connecting to Google's DNS.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        # print(ex)
        return False


def _pull_puzzle_input(day, seperator, cast):
    """
    Pull the puzzle data from the AOC website.

    :param day: (int, str) the AoC day puzzle input to fetch or example puzzle string.
    :param seperator: (str) A string separator to pass into str.split when consuming the puzzle data.
    :param cast: (None,type) A Python function often a type cast (int, str, lambda) to be run against each data element.

    :return: tuple of the data.
    """
    global _work, _offline, _code_path

    if _offline:
        with open(_code_path + r"\{}\day{}.txt".format(_year, day)) as file_handler:
            data_list = file_handler.read().split(seperator)
    elif type(day) is str:
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
            text = resp.text.strip("\n")
            if resp.ok:
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
    :param day: (int, str) the AoC day puzzle input to fetch or an example string
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
        if type(day) is int:  # only save the full puzzle data to the pickle file.
            puzzle_dict[day] = puzzle_input
            pickle.dump(puzzle_dict, open(_code_path + r'\{}\input.p'.format(_year), 'wb'))
    return puzzle_input


def _day1():
    """
    """
    day = "R5, L5, R5, R3"
    day = "R2, R2, R2"
    day = "R8, R4, R4, R8"
    day = 1
    instructions = get_input(day, ', ', None)
    right = {(0,1):(1,0), (1,0):(0,-1), (0,-1):(-1,0), (-1,0):(0,1)}
    left = {(0,1):(-1,0), (-1,0):(0,-1), (0,-1):(1,0), (1,0):(0,1)}
    decode = {(0,1):"North", (1,0):"East", (0,-1):"South", (-1,0):"West"}
    current_direction = np.array((0,1))
    position = np.array((0,0))
    intersections = {(0,0):None}
    found = False
    for instruction in instructions:
        instruction = instruction.strip()
        steps = int(instruction[1:])
        old_dir = current_direction
        if instruction[0] == "R":
            current_direction = np.array(right[tuple(current_direction)])
        elif instruction[0] == "L":
            current_direction = np.array(left[tuple(current_direction)])
        else:
            raise Exception(f"What should we do about {instruction[0]}")
        for step in range(steps):
            position += current_direction
            if tuple(position) in intersections.keys() and found is False:
                print(f"First location visited twice is {position} and is {abs(position).sum()} blocks away")
                found = True
            else:
                intersections[tuple(position)]=None
        #print(f"{instruction} {decode[tuple(old_dir)]}->{decode[tuple(current_direction)]} {position}")
    print(f"You are now {abs(position).sum()} blocks away")


def _day2():
    day = "ULL\nRRDDD\nLURDL\nUUUUD"
    day = 2
    instructions = get_input(day, "\n", None)
    move = {"U":(-1,0), "D":(1,0), "L":(0,-1), "R":(0,1)}
    # Part 1 keypad and start position
    position = np.array((1,1))
    keypad = np.array([[1,2,3,0,0],[4,5,6,0,0],[7,8,9,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    # Part 2 keypad and start position
    #position = np.array((2,0))
    #keypad = np.array([[0,0,1,0,0],[0,2,3,4,0],[5,6,7,8,9],[0,"A","B","C",0],[0,0,"D",0,0]])
    valid = []
    for i in range(5):
        for j in range(5):
            if keypad[(i,j)] not in ["0", 0]:
                valid.append((i,j))
    code = ""
    for instruction in instructions:
        for letter in instruction:
            new_position = position + np.array(move[letter])
            if tuple(new_position) in valid:
                position = new_position
        code += f"{keypad[tuple(position)]}"
        #print(f"{letter} {tuple(position)} {keypad[tuple(position)]}")
    print(f"The code is {code}")


def _day3():
    """
    Triangles
    """
    day=" 5 10 25\n 5 26 25\n2 2 3"
    day = 3
    design_docs = get_input(day, "\n", None)
    possible_triangles = 0
    for triangle in design_docs:
        dimensions = list(map(int, triangle.split()))
        if dimensions[0] + dimensions[1] > dimensions[2] and \
           dimensions[0] + dimensions[2] > dimensions[1] and \
           dimensions[1] + dimensions[2] > dimensions[0]:
            possible_triangles += 1
    print(f"Part 1 the possible triangles are {possible_triangles}")
    possible_triangles = 0
    design_docs = list(design_docs)
    while len(design_docs) > 0:
        threes = np.zeros([3,3],int)
        for i in range(3):
            row = list(map(int, design_docs.pop().split()))
            threes[:,i]=row
        for i in range(3):
            dimensions = list(threes[i])
            if dimensions[0] + dimensions[1] > dimensions[2] and \
               dimensions[0] + dimensions[2] > dimensions[1] and \
               dimensions[1] + dimensions[2] > dimensions[0]:
                possible_triangles += 1
    print(f"Part 2 the possible triangles are {possible_triangles}")


def _room_decode(name):
    """
    """
    name_id, checksum = name.split("[")
    checksum = checksum.strip("]")
    sector_id = name_id.split("-")[-1]
    name_letters = name_id.strip(sector_id)
    letter_freq = {}
    for c in ascii_lowercase:
        freq = name_letters.count(c)
        if freq > 0:
            letter_freq.setdefault(freq, [])
            letter_freq[freq].append(c)
    return (letter_freq, sector_id, checksum)

def _day4():
    """
    """
    day = "aaaaa-bbb-z-y-x-123[abxyz]\na-b-c-d-e-f-g-h-987[abcde]\nnot-a-real-room-404[oarel]\ntotally-real-room-200[decoy]"
    day = 4
    kiosk = get_input(day, "\n", _room_decode, True)
    sector_id_sum = 0
    for room in kiosk:
        freq_dict, sector_id, checksum = room
        freq_keys = sorted(freq_dict.keys(), reverse=True)
        my_checksum = []
        for key in freq_keys:
            my_checksum += freq_dict[key]
        my_checksum = "".join(str(x) for x in my_checksum[:5])
        if my_checksum == checksum:
            sector_id_sum += int(sector_id)
    print(f"Part 1 sum of sector IDs is {sector_id_sum}")


def go(day=1):
    try:
        return eval("_day{}".format(day))
    except Exception as e:
        print(e)


import concurrent.futures
import time


#if __name__ == "__main__":
#    loop = asyncio.get_event_loop()
#    loop.run_until_complete(c_thread(loop))