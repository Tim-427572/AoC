import copy
import hashlib
import inspect
import itertools
import math
import pickle
import re
import socket
import statistics
import sys
import time
from collections import defaultdict
from functools import lru_cache
from os import path
from string import ascii_lowercase

import numpy as np
import requests

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
    name_letters = name_id.strip(sector_id).strip("-")
    letter_freq = {}
    for c in ascii_lowercase:
        freq = name_letters.count(c)
        if freq > 0:
            letter_freq.setdefault(freq, [])
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
            l_num = l_num % 26
            l_num = 26 if l_num == 0 else l_num
            result += chr(l_num + 0x60)
        decrypted += result + " "
    return decrypted.strip()


def _day4():
    """
    """
    day = "aaaaa-bbb-z-y-x-123[abxyz]\na-b-c-d-e-f-g-h-987[abcde]\nnot-a-real-room-404[oarel]\ntotally-real-room-200[decoy]"
    day = 4
    kiosk = get_input(day, "\n", _room_decode, False)
    sector_id_sum = 0
    for room in kiosk:
        name, freq_dict, sector_id, checksum = room
        freq_keys = sorted(freq_dict.keys(), reverse=True)
        my_checksum = []
        for key in freq_keys:
            my_checksum += freq_dict[key]
        my_checksum = "".join(str(x) for x in my_checksum[:5])
        if my_checksum == checksum:
            sector_id_sum += int(sector_id)
            decrypted = _decrypt(name, int(sector_id))
            if "north" in decrypted:
                part_2_str = f"Part 2 The sector ID of the {decrypted} is {sector_id}"
    print(f"Part 1 sum of sector IDs is {sector_id_sum}")
    print(part_2_str)


def _day5(door_id):
    """
    """
    passcode1 = ""
    passcode2 = [None] * 8
    i = 0
    while True:
        #print(i)
        if len(passcode1) == 8 and None not in passcode2:
            break
        h = hashlib.md5(f"{door_id}{i}".encode("utf-8")).hexdigest()
        if h.startswith("00000"):
            passcode1 += h[5] if len(passcode1) < 8 else ""
            position = int(h[5],16)
            if position in range(0,8) and passcode2[position] == None:
                passcode2[position] = h[6]
            # print(passcode1)
            # print(passcode2)
        i += 1
    print(f"Part 1 passcode is {passcode1}")
    print(f"Part 2 passcode is {''.join(passcode2)}")


def _day6():
    """
    """
    puzzle = ("eedadn\n",
              "drvtee\n",
              "eandsr\n",
              "raavrd\n",
              "atevrs\n",
              "tsrnev\n",
              "sdttsa\n",
              "rasrtv\n",
              "nssdts\n",
              "ntnada\n",
              "svetve\n",
              "tesnvt\n",
              "vntsnd\n",
              "vrdear\n",
              "dvrsen\n",
              "enarar\n")
    puzzle = get_input(6, "\n", None)
    puzzle_list = []
    for line in puzzle:
        puzzle_list.append(list(line.strip()))
    message = np.array(puzzle_list, str)
    result1 = ""
    result2 = ""
    for column in message.T:
        values, counts = np.unique(column, return_counts=True)
        result1 += values[np.argmax(counts)]
        result2 += values[np.argmin(counts)]
    print(f"Part 1 the message was {result1}")
    print(f"Part 2 the message was {result2}")

        
def _day7(example=False):
    """
    IPv7
    """
    day = 7 if not example else ("abba[mnop]qrst\n"
                                 "abcd[bddb]xyyx\n"
                                 "aaaa[qwer]tyui\n"
                                 "ioxxoj[asdfgh]zxcvbn\n")
    puzzle = get_input(day, "\n", None)
    inside_re = re.compile("\[(.*?)\]")
    outside_re = re.compile("([^[\]]+)(?:$|\[)")
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
                inside = inside[1:]
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
                    outside = outside[1:]
                if abba:
                    break
        if abba and not hypernet_abba:
            p1_count += 1
        #print(ip, abba, hypernet_abba, p1_count)
    print(p1_count)


def _day7_p2(example=False):
    """
    IPv7
    """
    day = 7 if not example else ("aba[bab]xyz\n"
                                 "xyx[xyx]xyx\n"
                                 "aaa[kek]eke\n"
                                 "zazbz[bzb]cdb\n")
    puzzle = get_input(day, "\n", None)
    inside_re = re.compile("\[(.*?)\]")
    outside_re = re.compile("([^[\]]+)(?:$|\[)")
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
                            outside = ""
                outside = outside[1:]
            if ssl:
                break
        if ssl:
            p2_count += 1
        #print(ip, ssl, p2_count)
    print(p2_count)

def print_np(array):
    """
    Dumb helper function to print numpy arrays.
    """
    for row in array:
        print("".join(row))


def day8(example=False, reload=False):
    if example:
        day = ("rect 3x2\n"
               "rotate column x=1 by 1\n"
               "rotate row y=0 by 4\n"
               "rotate column x=1 by 1\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    instructions = get_input(day, "\n", None, reload)
    screen = np.full((6,50), " ", dtype=np.str)
    for instruction in instructions:
        if "rect" in instruction:
            rows = int(instruction.split("x")[1])
            cols = int(instruction.split("x")[0].split()[1])
            screen[:rows,:cols] = "#"
        elif "column" in instruction:
            col = int(instruction.split("=")[1].split()[0])
            amount = int(instruction.split("by")[1].strip())
            screen[:, col] = np.roll(screen[:, col], amount)
        elif "row" in instruction:
            row = int(instruction.split("=")[1].split()[0])
            amount = int(instruction.split("by")[1].strip())
            screen[row] = np.roll(screen[row], amount)
        else:
            raise Exception(instruction)
    print(f"Part 1, {np.count_nonzero(screen=='#')} pixels are lit")
    print("Part 2 the code is:")
    print_np(screen)

def go(day=1):
    try:
        return eval("day{}".format(day))
    except Exception as e:
        print(e)


# import concurrent.futures
# import time


#if __name__ == "__main__":
#    loop = asyncio.get_event_loop()
#    loop.run_until_complete(c_thread(loop))
