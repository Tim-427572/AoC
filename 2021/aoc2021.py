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

# Advent of Code
# Never did spend the time to work out how to get oAuth to work so this code expects you to
# manually copy over your session cookie value.
# Using a web browser inspect the cookies when logged into the Advent of Code website.
# Copy the value from the "session" cookie into a text file called "session.txt"

# Constants
_code_path = r'c:\AoC'
_offline = False
_year = 2021


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

    :param day: (int) the AoC day puzzle input to fetch.
    :param seperator: (str) A string separator to pass into str.split when consuming the puzzle data.
    :param cast: (None,type) A Python function often a type cast (int, str, lambda) to be run against each data element.

    :return: tuple of the data.
    """
    global _work, _offline, _code_path

    if _offline:

        with open(_code_path + r"\{}\day{}.txt".format(_year, day)) as file_handler:
            data_list = file_handler.read().split(seperator)
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
    :param day: (int) the AoC day puzzle input to fetch.
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
    if override:
        try:
            puzzle_dict.pop(day)
        except Exception:
            pass
    try:
        puzzle_input = puzzle_dict[day]
    except Exception:
        puzzle_input = _pull_puzzle_input(day, seperator, cast)
        puzzle_dict[day] = puzzle_input
        pickle.dump(puzzle_dict, open(_code_path + r'\{}\input.p'.format(_year), 'wb'))
    return puzzle_input


def _day1():
    """
    Elf submarine sonar depths.
    """
    depths = get_input(1, '\n', int)
    # depths = [199, 200, 208, 210, 200, 207, 240, 269, 260, 263]  # Example data
    p1_inc = 0
    p2_inc = 0
    for i in range(0, len(depths) - 1):
        if depths[i + 1] > depths[i]:
            p1_inc += 1
        if sum(depths[i + 1:i + 4]) > sum(depths[i:i + 3]):
            p2_inc += 1
    print(f"Part 1: increased {p1_inc} times")
    print(f"Part 2: increased {p2_inc} times")


def _day1_v():
    """
    Elf submarine sonar depths.
    """
    depths = np.array([199, 200, 208, 210, 200, 207, 240, 269, 260, 263])  # Example data
    depths = np.array(get_input(1, '\n', int))
    increases = (np.diff(depths) > 0).sum()
    print(f"Part 1: increased {increases} times")
    depths = np.convolve(depths, np.ones(3), 'valid')
    increases = (np.diff(depths) > 0).sum()
    print(f"Part 2: increased {increases} times")


def _day2():
    """
    Elf submarine movement commands
    """
    commands = get_input(2, '\n', lambda a:(a.split(" ")[0], int(a.split(" ")[1])))
    commands = [('forward', 5), ('down', 5), ('forward', 8), ('up', 3), ('down', 8), ('forward', 2)]  # Example data
    command_decode = {"up":-1, "down":1}
    horizontal = 0
    p1_depth = 0
    p2_depth = 0  # This is used as the aim in part 2.
    for direction, value in commands:
        if direction == "forward":
            horizontal += value
            p2_depth += value * p1_depth  # p1_depth is the same as aim for part 2
        else:
            p1_depth += value * command_decode[direction]
    print(f"Part 1 - position={horizontal}, depth={p1_depth} answer={horizontal * p1_depth}")
    print(f"Part 2 - position={horizontal}, depth={p2_depth} answer={horizontal * p2_depth}")


def _day2_v():
    """
    Elf submarine movement commands
    """
    commands = [('forward', 5), ('down', 5), ('forward', 8), ('up', 3), ('down', 8), ('forward', 2)]  # Example data
    commands = get_input(2, '\n', lambda a:(a.split(" ")[0], int(a.split(" ")[1])))
    # Split the puzzle data into two separate numpy arrays.
    directions, values = list(zip(*commands))
    directions = np.array(directions)
    values = np.array(values)
    # Up is negative
    values[np.where(directions=="up")] *= -1
    # Solve part 1.
    horizontal = sum(values[np.where(directions=="forward")])
    depth = sum(values[np.where(directions!="forward")])
    print(f"Part 1 - horizontal={horizontal}, depth={depth} - answer = {horizontal*depth}")


def _day3():
    diag = ['00100', '11110', '10110', '10111', '10101', '01111', '00111', '11100', '10000', '11001', '00010', '01010']
    #diag = list(get_input(3, '\n', None))
    num_size = len(diag[0])
    diag = list(map(lambda a:int(a,2), diag))
    gamma = epsilon = 0
    for i in range(num_size):
        digits = list(map(lambda a:(a>>i) & 1, diag))
        if digits.count(1) > digits.count(0):
            gamma |= 1 << i
        else:
            epsilon |= 1 << i
    print(f"Part 1 gamma={gamma}, epsilon={epsilon} answer = {gamma*epsilon}")
    oxygen = diag.copy()
    co2 = diag.copy()
    for i in range((num_size-1),-1,-1):
        if len(oxygen) == 1:
            break
        # print(f"Checking {i} digit")
        temp = oxygen.copy()
        digits = list(map(lambda a:(a>>i) & 1, oxygen))
        if digits.count(1) >= digits.count(0):
            # print("more 1s")
            remove = 0
        else:
            remove = 1
        for num in oxygen:
            if (num>>i) & 1 == remove:
                # print(f" removing {num:b}")
                temp.remove(num)
        oxygen = temp.copy()
    # print(f"Oxygen = {oxygen[0]}")
    for i in range((num_size-1),-1,-1):
        if len(co2) == 1:
            break
        #print(f"Checking {i} digit")
        temp = co2.copy()
        digits = list(map(lambda a:(a>>i) & 1, co2))
        if digits.count(0) <= digits.count(1):
            remove = 1
        else:
            remove = 0
        for num in co2:
            if (num>>i) & 1 == remove:
                #print(f" removing {num:05b}")
                temp.remove(num)
        co2 = temp.copy()
    #print(f"CO2 = {co2[0]}")
    print(f"Part 2 O2={oxygen[0]}, CO2={co2[0]}, answer = {oxygen[0] * co2[0]}")


def _day3_np():
    # Setup massaging the data
    data = ['00100', '11110', '10110', '10111', '10101', '01111', '00111', '11100', '10000', '11001', '00010', '01010']
    data = list(get_input(3, '\n', None))
    data = list(map(list, data))
    number_size = len(data[0])  # Store off the binary value number of digits
    data = np.array(data)

    # Part 1
    gamma = epsilon = 0
    diag = np.fliplr(data)  # Flip the array so that we walk lsb to msb
    for i in range(number_size):
        if np.count_nonzero(diag[0:, i:i+1] == "1") >= diag.shape[0] // 2:
            gamma |= 1 << i
        else:
            epsilon |= 1 << i
    print(f"Part 1 gamma={gamma}, epsilon={epsilon} answer = {gamma*epsilon}")

    # Part 2
    # Two copies of the array, one for the O2 generator test the other for the CO2 scrubber.
    oxygen = np.copy(data)
    co2 = np.copy(data)
    for i in range(number_size):  # Walk the columns
        if oxygen.shape[0] == 1:  # Stop when one number is left
            break
        # O2 keep the more common, if  1's and 0's are equal keep the 1's
        remove = "0" if np.count_nonzero(oxygen[0:, i:i+1] == "1") >= np.count_nonzero(oxygen[0:, i:i+1] == "0") else "1"
        oxygen = np.delete(oxygen, np.where(oxygen[0:, i:i+1] == remove)[0], axis=0)
    o2_gen_rating = int("".join(oxygen.tolist()[0]), 2)  # Messy way to convert the np array back into an int.
    for i in range(number_size):
        if co2.shape[0] == 1:
            break
        # CO2 keep the less common, if 1's and 0's are equal keep the 0's
        remove = "0" if np.count_nonzero(co2[0:, i:i+1] == "1") < np.count_nonzero(co2[0:, i:i+1] == "0") else "1"
        co2 = np.delete(co2, np.where(co2[0:, i:i+1] == remove)[0], axis=0)
    co2_scrub_rating = int("".join(co2.tolist()[0]), 2)
    print(f"Part 2 O2={o2_gen_rating}, CO2={co2_scrub_rating}, answer = {o2_gen_rating * co2_scrub_rating}")


def _day4():
    # puzzle data manipulation into a list and an np array
    bingo = ["7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1",
             "",
             "22 13 17 11  0",
             " 8  2 23  4 24",
             "21  9 14 16  7",
             " 6 10  3 18  5",
             " 1 12 20 15 19",
             "",
             " 3 15  0  2 22",
             " 9 18 13 17  5",
             "19  8  7 25 23",
             "20 11 10 24  4",
             "14 21 16 12  6",
             "",
             "14 21 17 24  4",
             "10 16 15  9 19",
             "18  8 23 26 20",
             "22 11 13  6  5",
             " 2  0 12  3  7"]
    #bingo = list(get_input(4, '\n', None))
    drawing = list(map(int, bingo.pop(0).split(",")))
    card_list = []
    card = []
    for line in bingo:
        if line == "":  # Start of the next card
            if card != []:
                card_list.append(card)
            card = []
        else:
            card.append(list(map(int, line.split())))
    if card != []:
        card_list.append(card)
    card_array = np.array(card_list)
    test_set = (-1, -1, -1, -1, -1)
    winning_board = None
    for ball in drawing:
        print(f" drawing ball {ball}")
        card_array[card_array == ball] = -1
        if np.where((card_array == test_set).all(axis=1))[0]:
            winning_board = np.where((card_array == test_set).all(axis=1))[0][0]
            break
        if np.where((card_array == test_set).all(axis=2))[0]:
            winning_board = np.where((card_array == test_set).all(axis=2))[0][0]
            break
    card_array[card_array == -1] = 0
    score = np.sum(card_array[winning_board]) * ball
    print(f"Part 1 winning board {winning_board} final score was {score}")
    print()
    card_array = np.array(card_list)
    test_set = (-1, -1, -1, -1, -1)
    winning_board = None
    last_winning_board = np.array([])
    last_ball = None
    for ball in drawing:
        print(f"drawing ball {ball}")
        card_array[card_array == ball] = -1
        #print(card_array)
        vert = np.where((card_array == test_set).all(axis=1))[0]
        horiz = np.where((card_array == test_set).all(axis=2))[0]
        #print(f"{vert}={vert.size}, {horiz}={horiz.size}")
        if horiz.size:
            for winning_board in horiz:
                if card_array.shape[0] > 0:
                    print(f"Removing board {winning_board}")
                    last_winning_board = np.copy(card_array[winning_board])
                    last_ball = ball
                    print(last_winning_board)
                    #card_array = np.delete(card_array, winning_board, 0)
                    card_array[winning_board] = [np.nan]
                    print(f"card_array.shape = {card_array.shape}")
                else:
                    break
        if vert.size:
            for winning_board in vert:
                if card_array.shape[0] > 0:
                    print(f"Removing board {winning_board}")
                    last_winning_board = np.copy(card_array[winning_board])
                    last_ball = ball
                    print(last_winning_board)
                    #card_array = np.delete(card_array, winning_board, 0)
                    card_array[winning_board] = [np.nan]
                    print(f"card_array.shape = {card_array.shape}")
                else:
                    break
    last_winning_board[last_winning_board == -1] = 0
    score = np.sum(last_winning_board) * last_ball
    print(f"Part 2 last winning board {winning_board} final score was {score}")
    #print(card_array)


def _day5():
    """
    """
    vents = ["0,9 -> 5,9",
             "8,0 -> 0,8",
             "9,4 -> 3,4",
             "2,2 -> 2,1",
             "7,0 -> 7,4",
             "6,4 -> 2,0",
             "0,9 -> 2,9",
             "3,4 -> 1,4",
             "0,0 -> 8,8",
             "5,5 -> 8,2"]
    vents = get_input(5, '\n', None)
    # Data manipulation
    vent_list = []
    for vent in vents:
        points = vent.split(" -> ")
        line = []
        for point in points:
            x, y = point.split(",")
            line.append((int(x),int(y)))
        vent_list.append(line)

    # Part 1
    # Quick check for map size
    max_x = max_y = 0
    for vent in vent_list:
        for point in vent:
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])
    sea_floor = np.zeros((max_x + 1, max_y + 1), np.int32)
    for vent in vent_list:
        if vent[0][0] == vent[1][0] or vent[0][1] == vent[1][1]:  # Straight lines
            start = (min(vent[0][0],vent[1][0]), min(vent[0][1], vent[1][1]))
            end = (max(vent[0][0],vent[1][0]) + 1, max(vent[0][1], vent[1][1]) + 1)
            sea_floor[start[0]:end[0], start[1]:end[1]] += 1
            #print(f"Points {vent[0]} -> {vent[1]}")
            #print(f"sea_floor[{start[0]}:{end[0]}, {start[1]}:{end[1]}] += 1")
            #print(sea_floor)
    print(f"Part 1 number of overlapping lines was {np.count_nonzero(sea_floor > 1)}")
    sea_floor = np.zeros((max_x + 1, max_y + 1), np.int32)
    for vent in vent_list:
        if vent[0][0] == vent[1][0] or vent[0][1] == vent[1][1]:  # Straight lines
            start = (min(vent[0][0],vent[1][0]), min(vent[0][1], vent[1][1]))
            end = (max(vent[0][0],vent[1][0]) + 1, max(vent[0][1], vent[1][1]) + 1)
            sea_floor[start[0]:end[0], start[1]:end[1]] += 1
            #print(f"Points {vent[0]} -> {vent[1]}")
        else: # should be some cool way to do this in np but the .diagonal method does not seem to let you write back.
            # Boring loop
            print(f"Diagonal points {vent[0]} -> {vent[1]}")
            start = vent[0] if vent[0][0] < vent[1][0] else vent[1]
            end = vent[0] if vent[0][0] > vent[1][0] else vent[1]
            slope = 1 if end[1] > start[1] else -1
            y = start[1]
            for x in range(start[0], end[0] + 1):
                #print(f"Point = ({x},{y})")
                sea_floor[x, y] += 1
                y += slope
            #print(f"sea_floor[{start[0]}:{end[0]}, {start[1]}:{end[1]}] += 1")
        #print(sea_floor)
    print(f"Part 2 number of overlapping lines was {np.count_nonzero(sea_floor > 1)}")


def _points_on_line(start, end):
    """
    Take two points and return a list of points on that line
    according to the rules for day 5.
    """
    points = []
    orientation = None
    if start[1] == end[1]:  # Horizontal
        orientation = "horizontal"
        for x in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
            points.append((x, start[1]))
    elif start[0] == end[0]:  # Vertical
        orientation = "vertical"
        #print(f"Vertical {vent[0]} -> {vent[1]}")
        for y in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
            points.append((start[0], y))
    else:
        orientation = "diagonal"
        diag_start = start if start[0] < end[0] else end
        diag_end = start if start[0] > end[0] else end
        slope = 1 if diag_end[1] > diag_start[1] else -1
        y = diag_start[1]
        for x in range(diag_start[0], diag_end[0] + 1):
            points.append((x, y))
            y += slope
    return points, orientation


def _day5_hash():
    """
    Hydrothermal Vents using sets.
    """
    vents = ["0,9 -> 5,9",
             "8,0 -> 0,8",
             "9,4 -> 3,4",
             "2,2 -> 2,1",
             "7,0 -> 7,4",
             "6,4 -> 2,0",
             "0,9 -> 2,9",
             "3,4 -> 1,4",
             "0,0 -> 8,8",
             "5,5 -> 8,2"]
    vents = get_input(5, '\n', None, True)
    # Data manipulation
    vent_list = []
    for vent in vents:
        points = vent.split(" -> ")
        line = []
        for point in points:
            x, y = point.split(",")
            line.append((int(x),int(y)))
        vent_list.append(line)
    # Part 1
    p1_vent_set = set()
    p2_vent_set = set()
    p1_overlap_set = set()
    p2_overlap_set = set()
    for vent in vent_list:
        points, orientation = _points_on_line(*vent)
        # Walk the positions and check for overlaps
        for point in points:
            if point in p1_vent_set and orientation in ["horizontal", "vertical"]:
                p1_overlap_set.add(point)
            if point in p2_vent_set:
                p2_overlap_set.add(point)
        # Update the set of all vent points.
        p2_vent_set.update(points)
        if orientation in ["horizontal", "vertical"]:
            p1_vent_set.update(points)
    print(f"Part 1 overlapping positions: {len(p1_overlap_set)}")
    print(f"Part 2 overlapping positions: {len(p2_overlap_set)}")


def _day6_first_try():
    """
    Lanternfish!

    Just used a list to hold all of the fish but it was clear just after switching
    from the example to the puzzle data set that this was not going to work.
    Probably will end up running out of space. Also each list.index call was O(n)
    Adjusted the code to use sort instead of index and was able to complete the part 1
    calculation, part 2 is probably never going to happen.
    Next optimization was to just slice off the 0's fish from the front of the list and only add them
    back (and the babies after the decrement phase). With this the code gets to around day 140 in a 
    reasonable amount of time.
    Got to day 172 but Python had consumed 15.6GB of ram. Next optimization was to use np arrays and
    use int8 for each fish in the array.
    """
    foo=1
    fish = ["3,4,3,1,2"]
    fish = get_input(6, '\n', None)
    fish_list = fish[0].split(",")
    fish_list = list(map(int, fish_list))
    print(fish_list)
    fish_array = np.array(fish_list, dtype=np.int8)
    print(fish_array)
    for day in range(256):
        # Forward
        print(f"Day: {day+1} currently {fish_array.size:,} fish")
        print(" find 0's")
        zeros = np.count_nonzero(fish_array == 0)
        fish_array[fish_array == 0] = 7
        print(" decrement")
        fish_array -= 1
        print(f" append {zeros:,} new fish")
        zero_array = np.array([8]*zeros, dtype=np.int8)
        fish_array = np.append(fish_array, zero_array)
        if day == 79:
            print(f"Part 1 number of fish: {fish_array.size}")
        # print(fish_array)
    print(f"Part 2 number of fish: {fish_array.size}")


def _day6():
    """
    Laternfish!
    Solution using list and taking advantage of pop/append to make the logic simple.
    """
    # Data manipulation
    fish_input = [3,4,3,1,2]
    fish_input = get_input(6, ',', int)

    # List to hold all of the different fish types.
    # The list index is the fish internal counter.
    # i.e. fish_list[1] will hold the number of fish with an internal timer of 1
    fish_list = [0]*9
    # Load the fish into the list
    for fish in fish_input:
        fish_list[fish] += 1
    # Puzzle solution
    for day in range(256):
        new_fish = fish_list.pop(0)  # pop the 0 fish, effectively decrements all the fish counters.
        fish_list.append(new_fish)  # New baby fish into the 8th index position
        fish_list[6] += new_fish  # Adjust the count of fish with internal timer 6 for all those that just reproduced.
        if day == 79:
            print(f"Part 1 after 80 days the number of fish: {sum(fish_list)}")
    print(f"Part 2 after 256 days the number of fish is: {sum(fish_list)}")


def _day6_dict():
    """
    Laternfish!
    Dictionary solution used to quickly solve part 2 for the leaderboard.
    """
    # Data manipulation
    fish = ["3,4,3,1,2"]
    fish = get_input(6, '\n', None)
    fish_list = fish[0].split(",")
    fish_list = list(map(int, fish_list))

    # Dictionary to hold all of the different fish types.
    fish_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
    # Load the fish into the dictionary
    for fish in fish_list:
        fish_dict[fish] += 1
    # Puzzle solution
    for day in range(256):
        new_fish = fish_dict[0]  # Store off the 0 fish for after the loop.
        for type in range(1,9):  # Decrement by moving down by one key.
            fish_dict[type-1] = fish_dict[type]
        fish_dict[8] = new_fish  # New baby fish
        fish_dict[6] += new_fish  # Fish that just reproduced
        if day == 79:
            print(f"Part 1 after 80 days the number of fish: {sum(fish_dict.values())}")
    print(f"Part 2 after 256 days the number of fish is: {sum(fish_dict.values())}")


def _coraline():
    """
    """
    example = [199, 200, 208, 210, 200, 207, 240, 269, 260, 263]
    example = get_input(1, '\n', int)
    times_it_increased = 0
    for index in range(len(example) - 1):
        print(f"The index was {index} checking {example[index]} against {example[index+1]}")
        if example[index + 1] > example[index]:
            times_it_increased = times_it_increased + 1
    print(f"Part 1 it increased {times_it_increased} times")


def _day7():
    """
    Crab submarines?
    """
    example = [16, 1, 2, 0, 4, 2, 7, 1, 2, 14]
    example = get_input(7, ',', int, True)
    # Part 1 find the point with the lowest fuel for all the ships (median)
    median = statistics.median(example)
    fuel = 0
    for position in example:
        fuel += abs(position - median)
    print(f"Part 1 median = {median}, fuel = {fuel}")
    # Part 2 each movement cost one additional fuel. Find the common point for all the ships.
    # The mean minimizes the sum of squares, we are doing the sum of n(n+1)/2.
    # The mean is not an exact match but should get us close.
    # Comes up with a fractional value so I was lazy and just tested both sides.
    min_fuel = 2**99
    mean = statistics.mean(example)
    print("Part 2")
    print(f" The mean was {mean}")
    means = [math.ceil(mean), math.floor(mean)]
    for mean in means:
        fuel = 0
        for position in example:
            # There is probably a more elegant way of doing this summation.
            # fuel += sum(range(abs(position - mean) + 1))
            # Ah found it, the math formula for a Triangular number is n*(n+1)/2
            movement = abs(position - mean)
            fuel += (movement * (movement + 1) // 2)
        print(f" position = {mean}, fuel = {fuel}")
        min_fuel = min(min_fuel, fuel)
    print(f"Part 2 the lowest fuel cost is {min_fuel}")





def maze_test():

    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (7, 6)

    path = a_star(maze, start, end)
    print(path)


def _remove(first, second):
    """
    Helper function to subtract the 7 segment display encodings.
    :param first: (str) The minuend
    :param second: (str) The subtrahend
    :return: (str) the difference
    """
    return "".join(c for c in first if c not in second)


def _check(first, second):
    """
    Helper function to check if the 7 segment encodings are equal
    given the fact the letters should match but the order could be different.
    """
    first_list = sorted(first)
    second_list = sorted(second)
    return first_list == second_list


"""
# Original decode tests.
for digit in line_list[i]:  # 3
    if len(digit) == 5 and _in(digit, decode_dict.get(1, "z")):
            decode_dict[3] = digit
for digit in line_list[i]:  # 2
    if len(digit) == 5 and len(_remove(digit, decode_dict.get(4, "z"))) == 3:
            decode_dict[2] = digit
for digit in line_list[i]:  # 5
    if len(digit) == 5 and 2 in decode_dict.keys() and len(_remove(digit, decode_dict.get(2, "z"))) == 2:
            decode_dict[5] = digit
for digit in line_list[i]:  # 9
    if len(digit) == 6 and 3 in decode_dict.keys() and len(_remove(digit, decode_dict.get(3, "z"))) == 1:
            decode_dict[9] = digit
for digit in line_list[i]:  # 6
    if len(digit) == 6 and 3 in decode_dict.keys() and\
       len(_remove(digit, decode_dict.get(3, "z"))) == 2 and \
       len(_remove(digit, decode_dict.get(1, "z"))) == 5:
            decode_dict[6] = digit
for digit in line_list[i]:  # 6
    if len(digit) == 6 and \
       len(_remove(digit, decode_dict.get(1, "z"))) == 4 and\
       len(_remove(digit, decode_dict.get(4, "z"))) == 3:
            decode_dict[0] = digit
"""


def _day8():
    """
    7-segment display decoding.
    """
    puzzle = """be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
            edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
            fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg
            fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb
            aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea
            fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb
            dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe
            bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef
            egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb
            gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce"""
    output_list = []
    line_list = []
    puzzle = puzzle.split('\n')
    puzzle = get_input(8, '\n', None)
    for line in puzzle:
        output_list.append(line.split(" | ")[1].split())
        line_list.append(line.replace(" | ", " ").split())
    count = 0
    for entry in output_list:
        for digit in entry:
            if len(digit) in [2,4,3,7]:
                count += 1
    print(f"Part 1 the unique output count was {count}")
    count = 0
    for i in range(len(line_list)):  # Walk through the entries
        decode_dict = {}
        # Pull out the unique digits
        for digit in line_list[i]:
            for c, l in [(1,2), (4,4), (7,3), (8,7)]: #1 has 2 segments, 4 has 4, 7 has 3, 8 has 7
                if len(digit) == l:
                    decode_dict[c] = digit
        # Time for the non-unique digits.
        # I checked the puzzle input and each test line always contains 1, 4, 7, and 8. Building my decode using 4 and 7.
        # This is kind of cheating I suppose and I could expand these rules if this was not the case.
        if 4 not in decode_dict.keys() or 7 not in decode_dict.keys():
            raise Exception("Either 4 or 7 was missing!")
        for digit in line_list[i]:
            if len(digit) == 6:  # 9, 6, or 0
                # Each of these check involve subtracting segments and seeing what is left based on the shapes of the characters.
                # For example if the character was 9 and you subtracted 4 and then 7 you would be left with only the 'g' segment left.
                if len(_remove(_remove(digit, decode_dict[4]), decode_dict[7])) == 1:  # _remove is just a string subtraction helper.
                    decode_dict[9] = digit
                if len(_remove(_remove(digit, decode_dict[4]), decode_dict[7])) == 2 and len(_remove(digit, decode_dict[7])) == 3:
                    decode_dict[0] = digit
                if len(_remove(_remove(digit, decode_dict[4]), decode_dict[7])) == 2 and len(_remove(digit, decode_dict[7])) == 4:
                    decode_dict[6] = digit
            if len(digit) == 5:  # 2, 5, or 3
                if len(_remove(digit, decode_dict[4])) == 3:
                    decode_dict[2] = digit
                if len(_remove(digit, decode_dict[4])) == 2 and len(_remove(digit, decode_dict[7])) == 2:
                        decode_dict[3] = digit
                if len(_remove(digit, decode_dict[4])) == 2 and len(_remove(digit, decode_dict[7])) == 3:
                        decode_dict[5] = digit

        # Should have a dictionary we can use for decoding.
        result = ""
        for digit in output_list[i]:
            found = False
            for k, v in decode_dict.items():
                if sorted(digit) == sorted(v):
                    result += f"{k}"
                    found = True
                    break
            if found is False:  # One additional check to make sure all of the digits decode.
                print()
                raise Exception(f"Error decoding {digit} using {decode_dict}")
        count += int(result)
    print(f"Part 2 sum of outputs was {count}")
        


def _day9():
    """
    """
    puzzle = [
        "2199943210",
        "3987894921",
        "9856789892",
        "8767896789",
        "9899965678"]
    puzzle = [
        "9999999",
        "9432129",
        "9321109",
        "9432109",
        "9999999"]
    #puzzle = get_input(9, '\n', None)
    puzzle1 = []
    for line in puzzle:
        puzzle1.append(list(map(int, list(line))))
    array = np.array(puzzle1)
    return array
    risk = 0
    low_points = 0
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            #print(f"{row},{col}")
            temp = array[max(row-1,0):min(row+2, array.shape[0]), max(col-1, 0):min(col+2, array.shape[1])]
            temp1 = temp - array[row,col]
            if np.count_nonzero(temp1 < 0):
                # Not a low point
                foo = 1
            else:
                print(f"low point at ({row},{col})\n{temp}")
                risk += array[row,col] + 1
                low_points += 1
    #np.set_printoptions(threshold=np.inf)
    #np.set_printoptions(linewidth=300)
    #np.set_printoptions(formatter={'all': lambda x: "{:d}".format(x)})
    print(f"Part 1 - {risk}")
    print(low_points)
    return
    array2 = np.copy(array)
    array2[array2 < 9] = 0
    #print(array2)
    basin = 1
    for row in range(array2.shape[0]):
        for col in range(array2.shape[1]):
            #print(f"{row},{col}")
            # Hmm need a better way to do this.
            if array2[row,col] == 9:
                continue
            up = array2[row-1,col] if row > 0 else 0
            down = array2[row+1,col] if row < array2.shape[0]-1 else 0
            left = array2[row,col-1] if col > 0 else 0
            right = array2[row,col+1] if col < array2.shape[1]-1 else 0
            if up in [0,9] and down in [0,9] and left in [0,9] and right in [0,9]:  # new basin
                array2[row,col] = basin
                basin += 1
            else:
                if up not in [0,9]:
                    array2[row,col] = up
                elif down not in [0,9]:
                    array2[row,col] = down
                elif left not in [0,9]:
                    array2[row,col] = left
                elif right not in [0,9]:
                    array2[row,col] = right
                else:
                    print(f"What happened? {array[row,col]} {up},{down},{left},{right}")
    # Cleanup, some edges like multiple 9's together cause trouble with my algorithm above.
    # This is a brute force cleanup phase to merge those problem areas.
    for i in range(20):
        for row in range(array2.shape[0]):
            for col in range(array2.shape[1]):
                #print(f"{row},{col}")
                # Hmm need a better way to do this.
                if array2[row,col] == 9:
                    continue
                this_basin = array2[row,col]
                up = array2[row-1,col] if row > 0 else 9
                down = array2[row+1,col] if row < array2.shape[0]-1 else 9
                left = array2[row,col-1] if col > 0 else 9
                right = array2[row,col+1] if col < array2.shape[1]-1 else 9
                if up not in [this_basin, 9]:
                    array2[array2==this_basin]=up
                elif down not in [this_basin, 9]:
                    array2[array2==this_basin]=down
                elif left not in [this_basin, 9]:
                    array2[array2==this_basin]=left
                elif right not in [this_basin, 9]:
                    array2[array2==this_basin]=right
                else:
                    foo=1
    #Size of the basin
    basin_list = []
    for b in range(basin,0,-1):
        if b == 9:
            continue
        basin_list.append(np.count_nonzero(array2==b))
    answer = 1
    for i in range(3):
        temp = max(basin_list)
        basin_list.remove(temp)
        answer *= temp

    print(answer)


def _day9_attempt2():
    """
    """
    puzzle = [
        "2199943210",
        "3987894921",
        "9856789892",
        "8767896789",
        "9899965678"]
    puzzle = get_input(9, '\n', None)

    vent_array = np.array(list(map(lambda x: list(map(int,x)), puzzle)))
    # Pad the array so that we don't have to worry about checking when at the border of the map.
    padded_vent_array = np.pad(vent_array, 1, 'constant', constant_values = 9)
    # Part 1 solution, collect the low points along the way.
    low_point_list = []
    basin_list = []
    risk = 0
    for row in range(1,vent_array.shape[0]+1):
        for col in range(1, vent_array.shape[1]+1):
            # np way of getting the up, down, left, right elements.
            rows = [row-1,row,row,row+1]
            cols = [col,col-1,col+1,col]
            if padded_vent_array[row, col] < np.min(padded_vent_array[(rows,cols)]):
                #print(f"Low point at ({row},{col})")
                risk +=  (padded_vent_array[row,col] + 1)  # Risk calculation for part 1
                # Part 2 code, put the low point in the input list to start
                input_list = [tuple([row,col])]
                basin_set = set()  # this set stores all the points in the basin.
                # While there is something left to check keep working.
                while input_list:
                    #print(input_list)
                    r, c = input_list.pop()
                    rows = [r-1, r,   r,   r+1]
                    cols = [c,   c-1, c+1, c]
                    if padded_vent_array[r,c] != 9:  # if the map position is not 9 put it in the set.
                        basin_set.add((r,c))
                    # Check the up, down, left, right elements and if they are not 9 put them into
                    # the input list to be processed on the next round.
                    for index in np.where(padded_vent_array[(rows,cols)] < 9)[0]:
                        if tuple([rows[index], cols[index]]) not in basin_set:
                            input_list.append(tuple([rows[index],cols[index]]))
                #print(f"Basin size was {len(basin_set)}")
                # Finished mapping the basin, add it to the list of basin sizes.
                basin_list.append(len(basin_set))
    print(f"Part 1 - risk is {risk}")
    print(f"Part 2 - multiply three largest basins to get {np.prod(sorted(basin_list, reverse=True)[:3])}")


def _day9_attempt3():
    """
    """
    from np.lib import stride_tricks
    puzzle = [
        "2199943210",
        "3987894921",
        "9856789892",
        "8767896789",
        "9899965678"]
    puzzle = get_input(9, '\n', None)
    vent_array = np.array(list(map(lambda x: list(map(int,x)), puzzle)))
    # Pad the map with mountains 9's to make testing easier.
    padded_vent_array = np.pad(vent_array, 1, 'constant', constant_values = 9)
    # Part 1 solution attempting to use np vectorization
    # Set up the data need to use the np stride to build a set of 3x3 arrays covering each point
    # in the map
    shape = (padded_vent_array.shape[0] - 2, padded_vent_array.shape[1] - 2, 3, 3)
    stride = padded_vent_array.strides * 2
    patches = stride_tricks.as_strided(padded_vent_array, shape=shape, strides=stride)
    # Still wish there was a better way to do this but use this command to reshape the 3x3 patches to
    # a 1x4 array holding up, down, left, right data.
    # axis 0 is the row, axis 1 is the column axis 2,3 are the 3x3 sub-arrays.
    adjacent = patches[:,:,[0,1,1,2],[1,0,2,1]]
    # Find the minimum for each patch and reshape the resulting data to match the original map.
    minimum_array = np.reshape(np.apply_over_axes(np.min, adjacent, 2), vent_array.shape)
    low_points = np.where(vent_array < minimum_array)
    print(f"Part 1 - The risk was {np.sum(vent_array[low_points]) + low_points[0].shape[0]}")


def _day10_orig():
    """
    """
    puzzle = [
        "[({(<(())[]>[[{[]{<()<>>",
        "[(()[<>])]({[<{<<[]>>(",
        "{([(<{}[<>[]}>{[]{[(<()>",
        "(((({<>}<{<{<>}{[]{[]{}",
        "[[<[([]))<([[{}[[()]]]",
        "[{[{({}]{}}([{[{{{}}([]",
        "{<[[]]>}<{[{[{[]{()[[[]",
        "[<(<(<(<{}))><([]([]()",
        "<{([([[(<>()){}]>(<<{{",
        "<{([{{}}[<[[[<>{}]]]>[]]"]
    puzzle = get_input(10, '\n', None)
    # Data provided in the puzzle description.
    reverse_dict = {"}":"{", "]":"[", ">":"<", ")":"("}
    forward_dict = {"{":"}", "[":"]", "<":">", "(":")"}
    p1_score_dict = {"}":1197, "]":57, ">":25137, ")":3}
    p2_score_dict = {")":1, "]":2, "}":3, ">":4}
    # Part 1
    p1_points = 0
    corrupted = []
    for line in puzzle:
        delimiter_list = []  # List to hold the open delimiters we have seen.
        for i in range(len(line)):  # Using the index to make it easier to check if we hit the end of the line.
            c = line[i]
            if c in forward_dict.keys():  # open delimiter
                delimiter_list.append(c)
            else:  # closing delimiter
                if delimiter_list[-1] != reverse_dict[c]: # If the last open delimiter does not match this close this is a problem.
                    if i != len(line)-1:
                        p1_points += p1_score_dict[c]
                        corrupted.append(line)
                        break
                else:  # The was a legal close delimiter.
                    delimiter_list.pop(-1)
    print(f"Part 1 - Total syntax error score is {p1_points}")
    # Could use a set difference but was feeling lazy.
    incomplete_list = []
    for line in puzzle:
        if line not in corrupted:
            incomplete_list.append(line)
    # Part 2 solution code.
    point_list = []
    for line in incomplete_list:
        points = 0
        delimiter_list = []
        for c in line:
            if c in forward_dict.keys():
                delimiter_list.append(c)
            else:  # closing delimiter
                if delimiter_list[-1] == reverse_dict[c]:  # Valid closing delimiter.
                    delimiter_list.pop()
                # Going to assume these are all incomplete and not corrupted.
        # Calculate the score.
        for c in reversed(delimiter_list):
            points *= 5
            points += p2_score_dict[forward_dict[c]]

        point_list.append(points)
    point_list.sort()
    # Get the middle score.
    print(f"Part2 - Middle score was {point_list[len(point_list)//2]}")


def _day10():
    """
    """
    puzzle = [
        "[({(<(())[]>[[{[]{<()<>>",
        "[(()[<>])]({[<{<<[]>>(",
        "{([(<{}[<>[]}>{[]{[(<()>",
        "(((({<>}<{<{<>}{[]{[]{}",
        "[[<[([]))<([[{}[[()]]]",
        "[{[{({}]{}}([{[{{{}}([]",
        "{<[[]]>}<{[{[{[]{()[[[]",
        "[<(<(<(<{}))><([]([]()",
        "<{([([[(<>()){}]>(<<{{",
        "<{([{{}}[<[[[<>{}]]]>[]]"]
    puzzle = get_input(10, '\n', None)
    # Data provided in the puzzle description.
    reverse_dict = {"}":"{", "]":"[", ">":"<", ")":"("}
    forward_dict = {"{":"}", "[":"]", "<":">", "(":")"}
    p1_score_dict = {"}":1197, "]":57, ">":25137, ")":3}
    p2_score_dict = {")":1, "]":2, "}":3, ">":4}
    # Solution using lists as a stack.
    p1_points = 0
    p2_point_list = []
    for line in puzzle:
        delimiter_list = []  # List to hold the open delimiters we have seen.
        line_points = 0
        for c in line:  # Using the index to make it easier to check if we hit the end of the line.
            if c in forward_dict.keys():  # Opening delimiter
                delimiter_list.append(c)
            else:  # Closing delimiter
                if delimiter_list[-1] != reverse_dict[c]: # If the last open delimiter does not match this close this is a problem.
                    p1_points += p1_score_dict[c]
                    break
                else:  # This was a legal close delimiter.
                    delimiter_list.pop()
        else:  # This was an incomplete line
            while len(delimiter_list) > 0:
                c = forward_dict[delimiter_list.pop()]
                line_points = line_points * 5 + p2_score_dict[c]
            p2_point_list.append(line_points)
    p2_point_list.sort()
    print(f"Part 1 - Total syntax error score is {p1_points}")
    print(f"Part2 - Middle score was {p2_point_list[len(p2_point_list)//2]}")  # Print the middle score in the p2_point_list

def flash(cavern):
    flash_set = set()
    while np.any(cavern>9):
        for point in zip(*np.where(cavern>9)):
                point = tuple(point)
                if point not in flash_set:
                    flash_set.add(point)
                    cavern[point]=0
                    cavern[point[0]-1:point[0]+2, point[1]-1:point[1]+2] += 1
    # Need to think up some more np'ish way of doing this.
    # Reset the flashed points to 0 again.
    for point in flash_set:
        cavern[point] = 0
    return len(flash_set)


def _day11():
    """
    """
    puzzle = ["5483143223",
              "2745854711",
              "5264556173",
              "6141336146",
              "6357385478",
              "4167524645",
              "2176841721",
              "6882881134",
              "4846848554",
              "5283751526"]
    # puzzle = ["11111", "19991", "19191", "19991", "11111"]
    puzzle = get_input(11, '\n', None)
    # Cavern transformation (pad with NaN to make indexing easy)
    cavern = np.array(list(map(lambda x: list(map(int,x)), puzzle)))
    all_flash_count = cavern.size
    cavern = np.pad(cavern, 1, "constant", constant_values=np.NaN)
    # Puzzle solution
    step = 1
    flash_count = 0
    p1_str = ""
    p2_str = ""
    while True:
        cavern += 1  # First energy level increases by 1
        # Then, any octopus flash
        flash_set = set()
        while np.any(cavern>9):
            for point in zip(*np.where(cavern>9)):
                    point = tuple(point)
                    if point not in flash_set:
                        flash_set.add(point)
                        # Slice the array to get the 3x3 sub-array (include diagonals this time).
                        cavern[point[0]-1:point[0]+2, point[1]-1:point[1]+2] += 1
            if flash_set:
                cavern[tuple(zip(*flash_set))] = 0  # Any octopus that flashed should stay at 0.
        # Track the puzzle status.
        step_flash_count = len(flash_set)
        flash_count += step_flash_count
        if step == 100:
            p1_str = f"Part 1 - {flash_count}"
        if step_flash_count == all_flash_count:
            p2_str = f"Part 2 - {step}"
            break  # Lazy while loop exit!
        step += 1
    print(p1_str)
    print(p2_str)


# A recursive DFS algorithm.
def _walk(paths_dict, path, visited, path_list, special):
    """
    """
    global count
    global path_set
    visited[path]=True
    if path == special[0]:
        special[1]+=1
    path_list.append(path)
    if path == "end":
        path_str = ",".join(path_list)
        if path_str not in path_set or True:
            path_set.add(path_str)
            count += 1
    else:
        for next_path in paths_dict[path]:
            if next_path.islower() and not visited.get(next_path, False) or next_path.isupper():
                _walk(paths_dict, next_path, visited, path_list,special)
            elif next_path == special[0] and special[1] <2:
                _walk(paths_dict, next_path, visited, path_list, special)
                
    path_list.pop()
    visited[path]=False
    if path == special[0]:
        special[1]-=1

count=0
path_set = set()  # Set to make sure we don't count duplicate paths.

def _day12():
    """
    Path through the caves.
    """
    global count
    global path_set
    puzzle = [ "start-A", "start-b", "A-c", "A-b", "b-d", "A-end", "b-end"]
    puzzle = [ "dc-end", "HN-start", "start-kj", "dc-start", "dc-HN", "LN-dc", "HN-end", "kj-sa", "kj-HN", "kj-dc",]
    #puzzle = [ "fs-end", "he-DX", "fs-he", "start-DX", "pj-DX", "end-zg", "zg-sl", "zg-pj", "pj-he", "RW-he", "fs-DX", "pj-RW", "zg-RW", "start-pj", "he-WI", "zg-he", "pj-fs", "start-RW",]
    puzzle = get_input(12, '\n', None)
    paths_dict = {}
    for path in puzzle:
        first, second = path.split("-")
        paths_dict.setdefault(first, [])
        paths_dict.setdefault(second, [])
        paths_dict[first].append(second)
        if first != "start" and second != "end":
            paths_dict[second].append(first)
    count = 0
    start_time = time.time()
    _walk(paths_dict, "start", {}, [], ["",0])  # Now to make part 1 work pass in an empty "special" cave.
    print(f"Part 1 - path count was {count}, time={time.time()-start_time}")
    # Make a list of the small caves so to walk through each time letting one be "special" and visited twice. 
    start_time = time.time()
    small_caves = []
    for cave in paths_dict.keys():
        if cave not in ["start","end"] and cave.islower():
            small_caves.append(cave)
    count = 0
    # Part 2 allows visiting one small cave twice. Iterate over the small caves and get all the paths.
    # store them in a set to avoid counting duplicates.
    path_set = set()
    for test_cave in small_caves:
        _walk(paths_dict, "start", {}, [], [test_cave,0])
    print(f"Part 2 - path count was {count}, time={time.time()-start_time}")


def _day13_orig():
    """
    Folding a sheet of paper using Numpy.
    My original solution had a bug on data sets where the array was not folded in "half"
    Added a fix to pad out the array that should work. I was lucky my data set didn't have this
    issue when attempting the solution at night.
    """
    puzzle = ["6,10", "0,14", "9,10", "0,3", "10,4", "4,11", "6,0", "6,12", "4,1", "0,13", "10,12", "3,4",
              "3,0", "8,4", "1,10", "2,14", "8,10", "9,0", "", "fold along y=7", "fold along x=5"]
    puzzle = get_input(13, '\n', None)
    point_list = []
    fold_list = []
    points_complete = False
    # Puzzle input processing.
    for line in puzzle:
        if line == "":
            points_complete = True
        elif not points_complete:
            col,row = line.split(",")
            point_list.append(tuple([int(row),int(col)]))
        elif points_complete:
            _,_,inst = line.split()
            dim, line = inst.split("=")
            fold_list.append(tuple([dim,int(line)]))
    # Make a correct sized array filled with 0's
    row_list, col_list = list(zip(*point_list))
    max_row = max(row_list) + 1
    max_col = max(col_list) + 1
    paper = np.zeros((max_row, max_col), np.int8)
    #paper = np.zeros((max_row+1, max_col+1), np.int8)
    # Fill in the data
    for point in point_list:
        paper[point] = 1
    # Start to solve the puzzle
    part1 = True
    for dim, line in fold_list:
        # In the puzzle y = rows so fold along the 0 (horizontal) axis in np, otherwise 1.
        axis = 0 if dim == "y" else 1
        if dim == "y":
            remainder = paper.shape[0] - line
            pad = abs((line*2)-paper.shape[0]+1)
            if remainder > line:
                padding = ((pad,0),(0,0))
            else:
                padding = ((0,pad),(0,0))
        else:
            remainder = paper.shape[1] - line
            pad = abs((line*2)-paper.shape[1]+1)
            if remainder > line:
                padding = ((0,0),(pad,0))
            else:
                padding = ((0,0),(0,pad))
        #print(f"shape:{paper.shape}, {dim}, fold_line:{line}, remainder:{remainder}, pad:{pad}, padding:{padding}")
        paper = np.pad(paper, padding, mode='constant', constant_values=0)
        paper = np.delete(paper,line, axis=axis)  # One line disappears into the crease of the paper.        
        #paper = np.delete(paper,line, axis=axis)  # One line disappears into the crease of the paper.
        folded_onto, folding = np.split(paper, 2, axis=axis)
        paper = folded_onto + np.flip(folding, axis=axis)
        if part1:
            print(f"Part 1 - There are {np.count_nonzero(paper)} dots")
            part1=False
    print("Part 2 - The code is:\n")
    # Pretty (readable) output. Need to make a np array printing helper function to keep around.
    # This is not the first time I've needed to print out a puzzle.
    for row in paper:
        row_str = ""
        for dot in row:
            if dot > 0:
                row_str += "#"
            else:
                row_str += " "
        print(row_str)


def _day13():
    """
    Folding transparent paper the right way. Only store the dots and don't make what is effectively a sparse
    array to try and hold them. Also messing around with making the print nicer.
    """
    puzzle = ["6,10", "0,14", "9,10", "0,3", "10,4", "4,11", "6,0", "6,12", "4,1", "0,13", "10,12", "3,4",
              "3,0", "8,4", "1,10", "2,14", "8,10", "9,0", "", "fold along y=7", "fold along x=5"]
    puzzle = get_input(13, '\n', None)
    # Puzzle input processing.
    row_list = []
    col_list = []
    fold_list = []
    for line in puzzle[:puzzle.index("")]:
        col,row = line.split(",")
        row_list.append(int(row))
        col_list.append(int(col))
    for line in puzzle[puzzle.index("")+1:]:
        _, _, inst = line.split()
        dim, line = inst.split("=")
        fold_list.append(tuple([dim,int(line)]))
    row_array = np.array(row_list)
    col_array = np.array(col_list)
    part_1 = True
    for dim, line in fold_list:
        if dim == "y":
            row_array = np.where(row_array >= line, np.abs(2*line-row_array), row_array)
        else:
            col_array = np.where(col_array >= line, np.abs(2*line-col_array), col_array)
        if part_1:
            temp = np.zeros((max(row_array)+1,max(col_array)+1))
            temp[(row_array,col_array)]=1
            print(f"Part 1 - After the first fold the number of visible dots was {np.count_nonzero(temp>0)}")
            part_1=False
    paper = np.empty((max(row_array)+1,max(col_array)+1),dtype="str")
    paper[:]=" "
    paper[(row_array,col_array)]=chr(0x2588)#"#"
    print()
    print(np.array2string(paper).replace("'","").replace("["," ").replace("]",""))


def _day14_orig():
    """
    Creating a polymer, this was my solution to part 1 which absolutely didn't scale past the example.
    """
    import collections
    puzzle = ["NNCB", "", "CH -> B", "HH -> N", "CB -> H", "NH -> C", "HB -> C", "HC -> B", "HN -> C", "NN -> C",
              "BH -> H", "NC -> B", "NB -> B", "BN -> B", "BB -> N", "BC -> B", "CC -> N", "CN -> C"]
    puzzle = get_input(14, '\n', None)
    initial = puzzle[0]
    polymer_dict = {}
    for transform in puzzle[2:]:
        poly_in, poly_out = transform.split(" -> ")
        polymer_dict[poly_in] = poly_out
    for step in range(1,41,1):
        print(step)
        output=""
        for i in range(len(initial)-1):
            #print(initial[i:i+2])
            temp = initial[i] + polymer_dict.get(initial[i:i+2],"") + initial[i+1]
            if len(output)>0:
                output = output[:-1]
            output += temp
        initial = output
        #print(initial)
    print(collections.Counter(initial).most_common(1)[0][1] - collections.Counter(initial).most_common()[-1][1])


def _day14_solution(steps=40):
    """
    Creating polymer pairs when the order within the polymer doesn't matter.
    Keep a dictionary of every pair in the polymer. For each reaction step subtract away the processed parts and add in the two new pairs.
    This was my inital solution but I'm not sure why it worked. There was a relationship between the count of the letters in
    the pairs and the count of the letters in the final answer.
    Also the code block to get minimum and maximum letters is kind of dumb. Should have just put them in a dictionary instead of a list.
    """
    puzzle = ["NNCB", "", "CH -> B", "HH -> N", "CB -> H", "NH -> C", "HB -> C", "HC -> B", "HN -> C", "NN -> C", "BH -> H",
              "NC -> B", "NB -> B", "BN -> B", "BB -> N", "BC -> B", "CC -> N", "CN -> C"]
    puzzle = get_input(14, '\n', None)
    initial = puzzle[0]  # Hard coding to grab the initial polymer value
    # Build a dictionary with the transformations.
    polymer_dict = {}
    for transform in puzzle[2:]:
        poly_in, poly_out = transform.split(" -> ")
        polymer_dict[poly_in] = poly_out
    # Going to store the polymer pairs in this dictionary and keep a count of how many times
    # the pair occurs (like the lanternfish).
    polymer_pairs = {}
    for i in range(len(initial)-1):
        polymer_pairs[initial[i:i+2]]=1

    # Need two copies of this dictionary, one to iterate and one to update.
    working_pairs = copy.copy(polymer_pairs)
    # Actual puzzle processing work, needed two rounds of optimization to get this to work.
    # Initially I had a loop here that ran 'counter' number of times incrementing the pairs each loop, this was a bad idea!
    start_time = time.perf_counter()
    for step in range(steps):
        # print(f"Working on {step}")
        for pair in polymer_pairs.keys():
            if pair in polymer_dict:
                # Found 'counter' number of this pair. Decrement in the working copy
                # Then add in the two new? pairs incrementing them by the counter.
                counter = polymer_pairs[pair]
                working_pairs[pair] -= counter
                working_pairs.setdefault(f"{pair[0]}{polymer_dict[pair]}",0)
                working_pairs[f"{pair[0]}{polymer_dict[pair]}"]+=counter
                working_pairs.setdefault(f"{polymer_dict[pair]}{pair[1]}",0)
                working_pairs[f"{polymer_dict[pair]}{pair[1]}"]+=counter
        # Copy the working dictionary back to the one we use for iteration.
        polymer_pairs = copy.copy(working_pairs)

    # Done, now take every letter in all of the dictionary keys and put it in this list to count occurrence.
    # All of the characters are capitalized so hardcoding a 0x40 subtraction.
    letters = [0]*37
    for key in polymer_pairs.keys():
        letters[ord(key[0])-0x40]+=polymer_pairs[key]
        letters[ord(key[1])-0x40]+=polymer_pairs[key]
    # Get the max and min character count
    max_letter = max(letters)
    # Too many 0's to just use the min function!
    min_letter=2**99
    for l in letters:
        if l > 0 and l < min_letter:
            min_letter=l
    # No idea why this works and I also had to manually round the answer up.
    # The /2 comes from looking at what my code came up with and comparing against the example answer for part 1.
    print(f"After {steps} steps the answer is {math.ceil(max_letter/2)-math.ceil(min_letter/2)}")
    #print(f"Processing time was {(time.perf_counter()-start_time)/(10**9):f}")


def _day14(steps=40):
    """
    Creating polymer pairs when the order within the polymer doesn't matter.
    Keep a dictionary of every pair in the polymer. For each reaction step increment the letter counter,
    subtract away the processed parts and add in the two new pairs.
    Slight refac
    Now with defaultdict, got tired of worrying about setting up the initial dictionaries for these puzzles.
    Also this time use a dictionary to count each inserted letter during the reaction which makes getting the
    final answer easy.
    """
    puzzle = ["NNCB", "", "CH -> B", "HH -> N", "CB -> H", "NH -> C", "HB -> C", "HC -> B", "HN -> C", "NN -> C", "BH -> H",
              "NC -> B", "NB -> B", "BN -> B", "BB -> N", "BC -> B", "CC -> N", "CN -> C"]
    puzzle = get_input(14, '\n', None)
    initial = puzzle[0]  # Hard coding to grab the initial polymer value
    
    # Build a dictionary with the transformations.
    polymer_dict = {}
    for transform in puzzle[2:]:
        poly_in, poly_out = transform.split(" -> ")
        polymer_dict[poly_in] = poly_out
    
    # Going to store the polymer pairs in this dictionary and keep a count of how many times
    # the pair occurs (like the lanternfish).
    polymer_pairs = defaultdict(int)
    for i in range(len(initial)-1):
        polymer_pairs[initial[i:i+2]]=1
    # Need a copies of this dictionary, one to iterate and one to update? Just didn't want to trip up using the same dict.
    working_pairs = copy.copy(polymer_pairs)

    # Dictionary to store a running letter count which starts with the initial polymer filled in.
    letter_dict = defaultdict(int)
    for l in initial:
        letter_dict[l]+=1

    # Actual puzzle processing work, needed two rounds of optimization to get this to work.
    # Initially I had a loop here that ran 'counter' number of times incrementing the pairs each loop, this was a bad idea!
    for step in range(steps):
        for pair in polymer_pairs.keys():
            if pair in polymer_dict:
                # Found n number of a pair. Decrement in the working copy (they are consumed)
                # Then add in the two (new?) pairs incrementing them by n
                counter = polymer_pairs[pair]
                working_pairs[pair] -= counter
                working_pairs[f"{pair[0]}{polymer_dict[pair]}"]+=counter
                working_pairs[f"{polymer_dict[pair]}{pair[1]}"]+=counter
                # Update the letter running tally so we can calculate the answer.
                letter_dict[polymer_dict[pair]] += counter
        # Copy the working dictionary back to the one we use for iteration.
        polymer_pairs = copy.copy(working_pairs)

    answer = max(letter_dict.values())-min(letter_dict.values())
    print(f"After {steps} steps the answer is {answer}")


class _Node():
    """
    Node for a A* path finding algorithm.
    """
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other_node):
        return self.position == other_node.position

def _a_star(maze, start, end):
    """
    A* path finding routine
    :param maze: ??
    :param start: (tuple) start position
    :param end: (tuple) end position
    :return: list of tuples as the path from start to end.
    """
    start_node = _Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = _Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []

    open_list.append(start_node)

    # Loop until we find the end node.
    while len(open_list) > 0:
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        # Pop the current node off the open list and put it onto the closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Check for the end node and walk backwards through the parents to build the path
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Reverse the path to make it traverse start to end.

        # Generate children phase
        children = []
        #for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            # Check if out of bounds.
            if node_position[0] not in range(len(maze)) or node_position[1] not in range(len(maze[len(maze) - 1])):
                continue
            # Check if passible position
            if maze[node_position[0]][node_position[1]] != 0:
                continue
            new_node = _Node(current_node, node_position)
            children.append(new_node)
        # Check children phase
        for child in children:
            for closed_child in closed_list:
                if child == closed_child:
                    continue
            child.g = current_node.g + 1  # Distance to start
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)  # heuristic
            child.f = child.g + child.h

            # Child is on the open list and the child f is greater than the open node.
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            open_list.append(child)


def _day15():
    """
    Mapping, tried implementing Dijkstra's algorithm a few different ways and they all failed to get
    the correct answer. The minimum cost path (MCP) is a common graph measurement. Found a python
    library that does exactly what the puzzle wanted and just installed it.
    """
    puzzle = ["1163751742", "1381373672", "2136511328", "3694931569", "7463417111",
              "1319128137", "1359912421", "3125421639", "1293138521", "2311944581", ]
    # pip install scikit-image
    from skimage.graph import route_through_array
    #np.set_printoptions(linewidth=300)
    #np.set_printoptions(threshold=sys.maxsize)
    puzzle = get_input(15, '\n', None)
    start_time = time.time()
    risk_list = []
    for line in puzzle:
        risk_list.append(list(map(int, list(line))))
    risk_array = np.array(risk_list)
    start = [0,0]
    end = [risk_array.shape[0]-1, risk_array.shape[1]-1]
    # Is it cheating to use a library that was designed to calculate the exact thing the puzzle wanted?
    # In order to disallow diagonal paths set fully_connected=False
    # To make the cost the sum of the values along the path set geometric=False
    route, cost = route_through_array(risk_array, start, end, fully_connected=False, geometric=False)
    print(f"Part 1 - The risk level was {int(cost)-risk_array[(0,0)]}")

    # Part 2 we need to expand the map, make a copy of the initial map.
    array_copy = np.copy(risk_array)
    # Expand the map in the columns direction (right) 4 more times, incrementing each time.
    for i in range(1,5):
        risk_array = np.append(risk_array, array_copy + i, axis=1)
    # Expand the map in the rows direction (down) 4 more times, incrementing each time.
    array_copy = np.copy(risk_array)
    for i in range(1,5):
        risk_array = np.append(risk_array, array_copy + i, axis=0)
    # The puzzle states that the values wrap at 9 back to 1. This is what I came up with there is probably a better math solution.
    # Tried using modulo and couldn't get it to work, that causes a wrap to 0 not 1.
    risk_array = np.where(risk_array > 9, risk_array - (risk_array//9)*9, risk_array)
    #print(a)
    # Now recalculate the costs again with this new risk map.
    end = [risk_array.shape[0]-1, risk_array.shape[1]-1]
    route, cost = route_through_array(risk_array, start, end, fully_connected=False, geometric=False)
    print(f"Part 2 - The risk level was {int(cost)-risk_array[(0,0)]}")
    print(f"Run time was {time.time()-start_time:0.4f}s")


class Sixteen:
    """
    This is a classed up version of my solution to day 16. Got silver on part 2 because of a
    bug related to parsing the I value. I was treating the 11b and 15b case as if both of them
    provided the number of sub-packets which was not true. This bug caused another in my code
    where I was trying to handle the fact I would get to the end of the BITS stream not having
    completed the sub-packets in the 15b case.
    """
    __whitespace = 0

    def __init__(self, **kwargs):
        self.stream = ""
        self.ip = 0
        self.out = []
        self.version_sum = 0
        self.load(kwargs.get("stream", ""))

    def _debug_print(self, s):
        if False:
            print(s)

    def _sub_packet(self, values):
            i = int(self.stream[self.ip:self.ip + 1], 2)
            sub_packet_length = 15 if i == 0 else 11
            self.ip += 1
            sub_packet_info = int(self.stream[self.ip:self.ip + sub_packet_length], 2)
            self.ip += sub_packet_length
            self.__whitespace += 2
            if i == 1:
                self._debug_print(f"{' '*self.__whitespace}sub-packet count = {sub_packet_info}")
                for i in range(sub_packet_info):
                    self._debug_print(f"{' '*self.__whitespace}sub-packet {i}")
                    values = self._decode(values)
            else:
                self._debug_print(f"{' '*self.__whitespace}sub-packets length = {sub_packet_info}")
                start = self.ip
                while self.ip - start < sub_packet_info:
                    self._debug_print(f"{' '*self.__whitespace}sub-packet - ip={self.ip}")
                    values = self._decode(values)
            #print(f"exit {pointer}, {values}")
            self.__whitespace-=2
            return values

    def _decode(self, values):
        # Header decode.
        version = int(self.stream[self.ip:self.ip + 3],2)
        self.version_sum += version
        self.ip +=3
        pkt_type = int(self.stream[self.ip:self.ip + 3],2)
        self.ip += 3
        # Special literal value packets
        if pkt_type == 4:
            number = ""
            continue_fetching = "1"
            while continue_fetching == "1":
                continue_fetching = self.stream[self.ip:self.ip + 1]
                self.ip += 1
                number += self.stream[self.ip:self.ip + 4]
                self.ip += 4
            number = int(number, 2)
            self._debug_print(f"{' '*self.__whitespace}Lit = {number}")
            values.append(number)
            return values
        else:  # all the other packets that perform math operations.
            command_values = []
            command_values = self._sub_packet(command_values)
            if pkt_type == 0:    # Sum
                self._debug_print(f"{' '*self.__whitespace}Sum - {values}")
                result = sum(command_values)
                debug_print(f"{' '*self.__whitespace}sum({command_values}) = {result}")
            elif pkt_type == 1:  # product
                self._debug_print(f"{' '*self.__whitespace}Mul - {values}")
                result = np.product(command_values)
                debug_print(f"{' '*self.__whitespace}prod({command_values}) = {result}")
            elif pkt_type == 2:  # minimum
                self._debug_print(f"{' '*self.__whitespace}Min {values}")
                result = min(command_values)
                self._debug_print(f"{' '*self.__whitespace}min({command_values}) = {result}")
            elif pkt_type == 3:  # maximum
                self._debug_print(f"{' '*self.__whitespace}Max {values}")
                result = max(command_values)
                self._debug_print(f"{' '*self.__whitespace}max({command_values}) = {result}")
            elif pkt_type == 5:  # greater than
                self._debug_print(f"{' '*self.__whitespace}GT - {values}")
                result = 1 if command_values[0] > command_values[1] else 0
                self._debug_print(f"{' '*self.__whitespace}{command_values[0]}>{command_values[1]}? => {result}")
            elif pkt_type == 6:  # less than
                self._debug_print(f"{' '*self.__whitespace}LT - {values}")
                result = 1 if command_values[0] < command_values[1] else 0
                self._debug_print(f"{' '*self.__whitespace}{command_values[0]}<{command_values[1]}? => {result}")
            elif pkt_type == 7:  # equal to
                self._debug_print(f"{' '*self.__whitespace}EQ - {values}")
                result = 1 if command_values[0] == command_values[1] else 0
                self._debug_print(f"{' '*self.__whitespace}{command_values[0]}=={command_values[1]}? => {result}")
            else:
                raise Exception("What just happened?")
            values.append(result)
            return values

    def run(self):
        """
        Start the decode of the BITS stream.
        return the first value in the output array.
        """
        if self.stream == "":
            raise Exception("No BITS stream to process")
        self.out = self._decode([])
        return self.out[0]

    def load(self, hex_string):
        """
        At first I was just casting the hex value to in int but ended up missing a leading 0.
        The quickest fix was just to consider each character independently and make sure they
        are 4 bits each.
        """
        for c in hex_string:
            self.stream += "{:04b}".format(int(c,16))


def _day16():
    """
    Parsing the BITS stream!
    """
    puzzle = "EE00D40C823060"
    puzzle = "0x8A004A801A8002F478"
    puzzle = "620080001611562C8802118E34"
    puzzle = "C0015000016115A2E0802F182340"
    puzzle = "A0016C880162017C3686B18A3D4780"
    puzzle = "C200B40A82"
    puzzle = "04005AC33890" # product
    puzzle = "880086C3E88112" # minimum == 7
    puzzle = "CE00C43D881120" # max == 9
    puzzle = "D8005AC2A8F0" # gt
    puzzle = "9C005AC2F8F0" # eq
    puzzle = "9C0141080250320F1802104A08" #?
    puzzle = get_input(16, '\n', None, True)[0]

    stream_decoder = Sixteen(stream=puzzle)
    result = stream_decoder.run()
    print(f"The sum of packet versions is {stream_decoder.version_sum}")
    print(f"The result of the instructions is {result}")


def fire(velocity, start, floor):
    """
    Calculate probe positions until it is below the target, no thrusters.
    """
    velocity = list(velocity)
    position = start
    point_list = []
    while position[1]>=floor:
        position[0] += velocity[0]
        position[1] += velocity[1]
        if velocity[0] != 0:
            velocity[0] = velocity[0] - 1 if velocity[0]>0 else velocity[0] + 1
        velocity[1] -= 1
        point_list.append(tuple(position))
    return point_list


def _day17():
    """
    Dumb loop searching setting up the ranges by hand.
    """
    target = [(20,30),(-10,-5)]
    target = [(153,199),(-114,-75)]
    floor = target[1][0]
    # Make a set of all the points in the target area
    target_set = []
    for x in range(target[0][0], target[0][1]+1):
        for y in range(target[1][0], target[1][1]+1):
            target_set.append((x,y))
    target_set = set(target_set)
    # Use a lazy loop to calculate the answers, picked the range by inspection of the initial puzzle values.
    max_y=0
    count = 0
    for x in range(0,200):
        for y in range(-200,200):
            point_list = fire((x,y),[0,0],floor)
            # print(f"({x},{y}) - {point_list}")
            if target_set.intersection(set(point_list)):
                count +=1
                for p in point_list:
                    if p[1]>max_y:
                        max_y=p[1]
                        print(f"New max y height using ({x},{y}), height={max_y}, current count of hits={count}")
    print(f"Unique initial velocity count = {count}")


def _add(a, b):
    """
    Add two snailfish numbers together.
    """
    #print("[" + a + "," + b + "] - Add two numbers, now start reduction.")
    return _reduce("[" + a + "," + b + "]")


def _explode(n):
    """
    The explode operation for snailfish numbers.
    """
    digits = "0123456789"
    new_n = n
    nest_level=0
    explode = False
    for index, value in enumerate(n):
        if value == "[":
            nest_level += 1
        elif value == "]":
            nest_level -= 1
            if nest_level >= 4:
                explode = True
                b_index = index
                r_close = index
                left = 0
                l_m=1
                right = 0
                r_m=1
                r=True
                while n[b_index] != "[":
                    c = n[b_index]
                    if c in digits:
                        if r:
                            right = right + (int(c) * r_m)
                            r_m *= 10
                        else:
                            left = left + (int(c) * l_m)
                            l_m *= 10
                    if c == ",":
                        r = False
                    b_index -= 1
                l_open = b_index
                a_left = False
                l_side = n[:l_open]
                r_side = n[r_close+1:]
                l_i = len(l_side)-1
                while l_i >= 0 and not a_left:  # Add to the left
                    c = l_side[l_i]
                    if c in digits:
                        new_c = c
                        end = l_i
                        l_i -= 1
                        while l_i >= 0:
                            c = l_side[l_i]
                            if c in digits:
                                new_c = c + new_c
                            else:
                                new_c = str(int(new_c)+left)
                                l_side = l_side[:l_i+1] + new_c + l_side[end+1:]
                                a_left = True
                                break
                            l_i -= 1
                    l_i -= 1
                a_right = False
                r_i =0
                while r_i < len(r_side) and not a_right:  # Add to the right
                    c = r_side[r_i]
                    if c in digits:
                        new_c = c
                        start = r_i
                        r_i += 1
                        while r_i < len(r_side):
                            c = r_side[r_i]
                            if c in digits:
                                new_c = new_c + c
                            else:
                                new_c = str(int(new_c) + right)
                                r_side = r_side[:start] + new_c + r_side[r_i:]
                                a_right = True
                                break
                            r_i += 1
                    r_i += 1
                new_n = l_side + "0" + r_side
                return explode, index, new_n, f"[{left},{right}]"
    return explode, index, new_n, ""


def _split(n):
    """
    Perform the snailfish split operation.
    """
    digits = "0123456789"
    split = False
    i=0
    while split is False and i < len(n):
        if n[i] in ",]":  # end of a number
            mul = 1
            num = 0
            b_index = i - 1
            while b_index >= 0:
                c = n[b_index]
                if c in digits:
                    num = num + (int(c) * mul)
                    mul *= 10
                elif c in ",[":
                    if num > 9:
                        # Split
                        start = b_index + 1
                        end = b_index
                        b_index += 1
                        while b_index < len(n):
                            if n[b_index] in ",[]":
                                end=b_index
                                break
                            b_index += 1
                        left = num // 2
                        right = num - left
                        new_n = n[:start] + f"[{left},{right}]" + n[end:]
                        return True, i, new_n, f"{num}"
                    break
                b_index -= 1
        i+=1
    return split, 0, n, f"{num}"


def _reduce(n):
    """
    Reduce the snailfish number.
    """
    while True:
        e, e_i, e_n, e_s = _explode(n)
        s, s_i, s_n, s_s = _split(n)
        if e:  # Explode always happens first.
            n = e_n
            #print(f"{n} - Explode {e_s} at index {e_i}")
        elif s:  # Split
            n = s_n
            #print(f"{n} - Split {s_s} at index {e_i}")
        else:
            break
    return n


def _magnitude(l):
    """
    Recursive function to calculate the magnitude of a snailfish number.
    
    """
    if type(l) is str:  # Transform the string into a python list using eval.
        l = eval(l)
    if len(l) == 1:  # Done, return the answer.
        return l
    elif len(l) == 2:  # Two things
        if type(l[0]) is int and type(l[1]) is int:  # Two ints
            return (l[0] * 3) + (l[1] * 2)
        elif type(l[0]) is int and type(l[1]) is not int:  # Int and a list (still need to process the right)
            right = _magnitude(l[1])
            return (l[0] * 3) + (right * 2)
        elif type(l[0]) is not int and type(l[1]) is int:  # Int and a list (still need to process the left)
            left = _magnitude(l[0])
            return (left * 3) + (l[1] * 2)
        else: # Two lists, process both.
            left = _magnitude(l[0])
            right = _magnitude(l[1])
            return left*3 + right*2
    else:  # More than two things, process the left and right and then call back.
        left = _magnitude(l[0])
        right = _magnitude(l[1])
        _magnitude([left,right])


def _day18():
    """
    """
    puzzle = [ '[1,1]', '[2,2]', '[3,3]', '[4,4]' ]
    puzzle = [ '[1,1]', '[2,2]', '[3,3]', '[4,4]', '[5,5]']
    puzzle = [ '[1,1]', '[2,2]', '[3,3]', '[4,4]', '[5,5]','[6,6]']
    puzzle = [  '[[[[4,3],4],4],[7,[[8,4],9]]]','[1,1]']
    puzzle = ["[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]", "[[[5,[2,8]],4],[5,[[9,9],0]]]", "[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]",
              "[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]", "[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]", "[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]",
              "[[[[5,4],[7,7]],8],[[8,3],8]]", "[[9,3],[[9,9],[6,[4,9]]]]", "[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]", "[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]"]
    puzzle = ["[[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]]", "[7,[[[3,7],[4,3]],[[6,3],[8,8]]]]", "[[2,[[0,8],[3,4]]],[[[6,7],1],[7,[1,6]]]]",
              "[[[[2,4],7],[6,[0,5]]],[[[6,8],[2,8]],[[2,1],[4,5]]]]", "[7,[5,[[3,8],[1,4]]]]", "[[2,[2,2]],[8,[8,1]]]",
              "[2,9]", "[1,[[[9,3],9],[[9,0],[0,7]]]]", "[[[5,[7,4]],7],1]", "[[[[4,2],2],6],[8,7]]"]
    puzzle = list(get_input(18,'\n',None))

    # Part 1
    total = puzzle.pop(0)
    for number in puzzle:
        total = _add(total, number)
        #print(f"{total} - Reduced")
    magnitude = _magnitude(total)
    print(f"Part 1 - Magnitude of the final sum is {magnitude}")
    # Part 2 - Super brute force search!
    max_magnitude = 0
    for a in puzzle:
        for b in puzzle:
            total = _add(a,b)
            magnitude = _magnitude(total)
            max_magnitude = magnitude if magnitude > max_magnitude else max_magnitude
            #print(max_magnitude)
    print(f"Part 2 - The maximum magnitude for any two numbers is {max_magnitude}")


def _flip(scanner_set, flip):
    """
    Change the "up" direction by reversing the sign of an axis (multiply with -1)
    """
    new_set = set()
    for point in scanner_set:
        new_set.add(tuple(np.array(point) * np.array(flip)))
    return new_set

def _rotate(scanner_set, rotate):
    """
    Swap axis in the data set. For example take the X coordinates and put them in the Y position
    and vise versa.
    """
    new_set = set()
    for point in scanner_set:
        temp_list=[]
        for i in rotate:
            temp_list.append(point[i])
        new_set.add(tuple(temp_list))
    return new_set


def _align(scanner_set, a, b):
    """
    Performs a linear 3d movement of the points in a set.
    """
    delta_x = a[0]-b[0]
    delta_y = a[1]-b[1]
    delta_z = a[2]-b[2]
    new_set = set()
    for x,y,z in scanner_set:
        n_x = x + delta_x
        n_y = y + delta_y
        n_z = z + delta_z
        new_set.add(tuple([n_x, n_y, n_z]))
    return new_set, (delta_x,delta_y,delta_z)


def _unwind(transforms, start, visited, path_list):
    """
    A DFS to find the path between the scanner "start" and scanner 0.
    """
    visited[start]=True
    path_list.append(start)
    if start == 0:
        return True
    else:
        for n in transforms:
            if n[0] == start and not visited.get(n[1], False):
                test = _unwind(transforms, n[1], visited, path_list)
                if test:
                    return test
    path_list.pop()
    visited[start]=False
    return False


def _search(scanner_1, scanner_2, scanner_dict):
    """
    Function to iterate through all of the different orientations.
    I'm testing 48 things here where the puzzle says there are 24.
    It is likely some transformations are duplicates but I don't want to think about it right now.
    """
    for rotate in itertools.permutations([0,1,2]):  # Rotate x->y or y->x, etc
        beacons_1 = copy.deepcopy(scanner_dict[scanner_2])
        beacons_2 = _rotate(beacons_1, rotate)
        for flip in itertools.product([1,-1],[1,-1],[1,-1]):  # Flip x *= -1 or y *= -1, etc.
            beacons_3 = _flip(beacons_2, flip)
            # Align pick a pair of points and shift one set by the difference between them.
            for a, b in itertools.product(scanner_dict[scanner_1], beacons_3):
                test_set, delta = _align(beacons_3, a, b)
                intersect = scanner_dict[scanner_1].intersection(test_set)
                if len(intersect) >= 12:
                    flip_str = f"({flip[0]:2d},{flip[1]:2d},{flip[2]:2d})"
                    delta_str = f"({delta[0]:5d},{delta[1]:5d},{delta[1]:5d})"
                    print(f"Scanner {scanner_1:2d} and {scanner_2:2d} overlap using rotation={rotate}, flip={flip_str}, delta={delta_str}")
                    probes = copy.deepcopy(test_set)
                    return f"{scanner_2},{scanner_1}", {"rotate":rotate, "flip":flip, "align":delta}, [scanner_2, scanner_1]
    return "", [], []


def _day19():
    """
    Scanners and Beacons.
    """
    puzzle = [
        "--- scanner 0 ---",
        "404,-588,-901", "528,-643,409", "-838,591,734", "390,-675,-793", "-537,-823,-458", "-485,-357,347", "-345,-311,381", "-661,-816,-575", "-876,649,763", "-618,-824,-621", "553,345,-567", "474,580,667", "-447,-329,318", "-584,868,-557", "544,-627,-890", "564,392,-477", "455,729,728", "-892,524,684", "-689,845,-530", "423,-701,434", "7,-33,-71", "630,319,-379", "443,580,662", "-789,900,-551", "459,-707,401", "",
        "--- scanner 1 ---",
        "686,422,578", "605,423,415", "515,917,-361", "-336,658,858", "95,138,22", "-476,619,847", "-340,-569,-846", "567,-361,727", "-460,603,-452", "669,-402,600", "729,430,532", "-500,-761,534", "-322,571,750", "-466,-666,-811", "-429,-592,574", "-355,545,-477", "703,-491,-529", "-328,-685,520", "413,935,-424", "-391,539,-444", "586,-435,557", "-364,-763,-893", "807,-499,-711", "755,-354,-619", "553,889,-390", "",
        "--- scanner 2 ---",
        "649,640,665", "682,-795,504", "-784,533,-524", "-644,584,-595", "-588,-843,648", "-30,6,44", "-674,560,763", "500,723,-460", "609,671,-379", "-555,-800,653", "-675,-892,-343", "697,-426,-610", "578,704,681", "493,664,-388", "-671,-858,530", "-667,343,800", "571,-461,-707", "-138,-166,112", "-889,563,-600", "646,-828,498", "640,759,510", "-630,509,768", "-681,-892,-333", "673,-379,-804", "-742,-814,-386", "577,-820,562", "",
        "--- scanner 3 ---",
        "-589,542,597", "605,-692,669", "-500,565,-823", "-660,373,557", "-458,-679,-417", "-488,449,543", "-626,468,-788", "338,-750,-386", "528,-832,-391", "562,-778,733", "-938,-730,414", "543,643,-506", "-524,371,-870", "407,773,750", "-104,29,83", "378,-903,-323", "-778,-728,485", "426,699,580", "-438,-605,-362", "-469,-447,-387", "509,732,623", "647,635,-688", "-868,-804,481", "614,-800,639", "595,780,-596", "",
        "--- scanner 4 ---",
        "727,592,562", "-293,-554,779", "441,611,-461", "-714,465,-776", "-743,427,-804", "-660,-479,-426", "832,-632,460", "927,-485,-438", "408,393,-506", "466,436,-512", "110,16,151", "-258,-428,682", "-393,719,612", "-211,-452,876", "808,-476,-593", "-575,615,604", "-485,667,467", "-680,325,-822", "-627,-443,-432", "872,-547,-609", "833,512,582", "807,604,487", "839,-516,451", "891,-625,532", "-652,-548,-490", "30,-46,-14", "",
    ]
    puzzle = get_input(19,'\n',None)

    # Process the input data putting each one in a set.
    scanner_dict = defaultdict(set)
    scanner = None
    for line in puzzle:
        if "---" in line:
            line_list = line.split()
            scanner = int(line_list[2])
        elif line == "":
            foo=1
        else:
            a,b,c = line.split(",")
            scanner_dict[scanner].add(tuple([int(a),int(b),int(c)]))

    # Do the work of finding pairs of scanners that overlap.
    transform_dict={}  # Dictionary to hold the transformation information between two scanners.
    transform_pairs = []  # List of graph connections used in the DFS
    # Begin a big brute force search.
    start_time = time.time()
    for scanner_1, scanner_2 in itertools.combinations(scanner_dict.keys(),2):  # Check every scanner against each other.
        key, value, transform = _search(scanner_1, scanner_2, scanner_dict)
        if key != "":  # Found 12 beacon overlatp.
            transform_dict[key]=value
            transform_pairs.append(transform)
            # We know these two overlap, get the transformation in the other direction
            # This could be done with some quick math but it was late when I was working on this.
            # It also only seems to add around 30s to the run time (which is slow because of the brute force search)
            key_1, value_1, transform_1 = _search(scanner_2, scanner_1, scanner_dict)
            transform_dict[key_1]=value_1
            transform_pairs.append(transform_1)
    search_time = (time.time()-start_time)/60.0
    print(f"The search took = {search_time:0.3f} min")
    # Use the Graph of transformations to answer the questions.
    beacon_set = set()
    beacon_set.update(scanner_dict[0])
    max_distance = 0
    scanner_positions = {}  # Dictionary to store the scanner positions for the Manhattan question.
    for scanner in scanner_dict.keys():
        transform_list=[]
        # Unwind returns a list of scanners that if we take the beacon info and transform it through each we will end up
        # in the scanner 0 reference frame (hard coded that we want to end up at scanner 0).
        u = _unwind(transform_pairs, scanner, {}, transform_list)
        if u:
            temp = copy.deepcopy(scanner_dict[transform_list[0]])
            z = [[0,0,0]]  # To find the scanner position (relative to scanner 0) take a 0,0,0 point and translate it as well.
            for i in range(len(transform_list)-1):
                step=f"{transform_list[i]},{transform_list[i+1]}"
                temp = _rotate(temp,transform_dict[step]["rotate"])
                temp = _flip(temp, transform_dict[step]["flip"])
                temp, _ = _align(temp, transform_dict[step]["align"], [0,0,0])
                z = _rotate(z,transform_dict[step]["rotate"])
                z = _flip(z, transform_dict[step]["flip"])
                z, _ = _align(z, transform_dict[step]["align"], [0,0,0])
            beacon_set.update(temp)
            scanner_positions[scanner]=z.pop()
        else:
            print(f"Attempt to map scanner {scanner} to scanner 0 failed")
    print(f"Part 1 - The number of beacons is {len(beacon_set)}")
    # Part 2
    max_distance = 0
    max_pair = []
    for scanner_1, scanner_2 in itertools.combinations(scanner_dict.keys(), 2):
        distance = np.sum(abs(np.array(scanner_positions[scanner_1])-np.array(scanner_positions[scanner_2])))
        if distance > max_distance:
            max_distance = distance
            max_pair = [scanner_1, scanner_2]
    print(f"Part 2 - The distance between {max_pair[0]} and {max_pair[1]} is {max_distance}")




def _day20_orig(real_data=False, steps=2):
    """
    Fixed up version of the code that I used to solve the puzzle when it unlocked.
    the loop over the patches was not the right way to do this. Originally I was
    using .join to convert the patch into a binary string then to a decimal value.
    """
    from numpy.lib import stride_tricks
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=500)
    puzzle = [
        #"#.#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..###..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###.######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#..#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#......#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#.....####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.......##..####..#...#.#.#...##..#.#..###..#####........#..####......#...",
        "..#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..###..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###.######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#..#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#......#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#.....####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.......##..####..#...#.#.#...##..#.#..###..#####........#..####......#..#",
        "",
        "#..#.",
        "#....",
        "##..#",
        "..#..",
        "..###"
    ]
    if real_data:
        puzzle = list(get_input(20, '\n', None, False))
    # Massage the data to get a 0's and 1 arrays instead of # and .
    enhancement = puzzle.pop(0)
    f=puzzle.pop(0)
    enhancement = np.array(list(enhancement))
    enhancement[enhancement=="."]=0
    enhancement[enhancement=="#"]=1
    enhancement = np.asarray(enhancement, dtype=np.int8)
    l=[]
    for line in puzzle:
        line = line.replace(".","0")
        line = line.replace("#","1")
        l.append(list(map(int,list(line))))
    image = np.array(l)
    # Solution code.
    t=np.array([[2**8,2**7,2**6],[2**5,2**4,2**3],[2**2,2**1,2**0]], dtype=np.int32)  # multiply by this and sum to translate binary array to int.
    image = np.pad(image, 1, 'constant', constant_values=0)  # Initial padding.
    #image = np.pad(image, steps+5, 'constant', constant_values=0)  # The image can grow  outward by the number of steps.
    s_time = time.time()
    sys.stdout.write("Step   ")
    for step in range(steps):
        sys.stdout.write("\b\b\b")
        sys.stdout.write(f"{step+1:3d}")
        sys.stdout.flush()
        p_v = 0 if image[0,0] == 0 else 1  # The bug that was kicking me, when you pad use the correct value.
        image = np.pad(image, 2, 'constant', constant_values=p_v)  # Each processing step consumes 2 in each direction, so pad it.
        # Use the same numpy trick from the vent tube risk one.
        shape = (image.shape[0] - 2, image.shape[1] - 2, 3, 3)
        stride = image.strides * 2
        # Patches is a 4D array holding all the 3x3 things we need to process.
        patches = stride_tricks.as_strided(image, shape=shape, strides=stride)
        # There is probably a better numpy way to do this, need to revisit this.
        # Build the new image out of the enhancement array.
        next_image = []
        for row in range(patches.shape[0]):
            r=[]
            for col in range(patches.shape[1]):
                append_value = np.copy(enhancement[np.sum(patches[row,col] * t)])
                r.append(append_value)
            next_image.append(r)
        image = np.array(next_image)
        # if False:  # Routine to print a pretty array, for debug.
        #     output = np.empty(image.shape,dtype="str")
        #     output[:]="."
        #     output[np.where(image==1)]="#"
        #     print()
        #     print(np.array2string(output).replace("'","").replace("["," ").replace("]",""))
    sys.stdout.flush()
    #output = np.empty(image.shape,dtype="str")
    #output[:]="."
    #output[np.where(image==1)]="#"
    #print()
    #print(np.array2string(output).replace("'","").replace("["," ").replace("]",""))
    print()
    print(f"After {steps} enhancements the number of lit pixels is {np.count_nonzero(image)}")
    print(f"time = {time.time()-s_time}")


def _day20(real_data=False, steps=2):
    """
    """
    from numpy.lib import stride_tricks
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=500)
    puzzle = [
        "..#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..###..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###.######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#..#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#......#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#.....####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.......##..####..#...#.#.#...##..#.#..###..#####........#..####......#..#",
        "",
        "#..#.",
        "#....",
        "##..#",
        "..#..",
        "..###"
    ]
    if real_data:
        puzzle = list(get_input(20, '\n', None, False))
    # Massage the data to get a 0's and 1 arrays instead of # and .
    enhancement = puzzle.pop(0)
    puzzle.pop(0)  # Remove the empty line.
    enhancement = np.array(list(enhancement))
    enhancement[enhancement=="."]=0
    enhancement[enhancement=="#"]=1
    enhancement = np.asarray(enhancement, dtype=np.int8)
    l=[]
    image = np.empty((0,len(puzzle[0])),dtype=np.str)
    for line in puzzle:
        image = np.append(image, [list(line)], axis=0)
    image[image=="."]=0  # Replace . with 0 and # with 1.
    image[image=="#"]=1
    image = np.asarray(image, dtype=np.int32)  # cast it back to an integer numpy array.

    # Solution code.
    t=np.array([[2**8, 2**7, 2**6],[2**5, 2**4, 2**3],[2**2, 2**1, 2**0]])  # multiply by this and sum to translate binary array to int.
    image = np.pad(image, 1, 'constant', constant_values=0)  # Initial padding.
    for step in range(steps):
        p_v = 0 if image[0,0] == 0 else 1  # The bug that was kicking me, when you pad use the correct value.
        image = np.pad(image, 2, 'constant', constant_values=p_v)  # Each processing step consumes 2 in each direction, so pad it.
        # Use the same numpy trick from the vent tube risk one.
        shape = (image.shape[0] - 2, image.shape[1] - 2, 3, 3)
        stride = image.strides * 2
        # Patches will be a 4D array holding all the 3x3 things we need to process.
        patches = stride_tricks.as_strided(image, shape=shape, strides=stride)
        enhancement_index = np.sum(patches*t, axis=(2,3))  # Sum the patch to get the enhancement index value.
        image = np.take(enhancement, enhancement_index)  # Take the value from enhancement based on the index in the new enhancement_index array
    #output = np.empty(image.shape,dtype="str")
    #output[:]="."
    #output[np.where(image==1)]="#"
    #print()
    #print(np.array2string(output).replace("'","").replace("["," ").replace("]",""))
    print(f"After {steps} enhancements the number of lit pixels is {np.count_nonzero(image)}")


class deterministic_die:
    def __init__(self):
        self.counter = 0
        self.value = 0
        self.result = []
        self.move = 0
    
    def roll(self):
        self.result = []
        self.move = 0
        for i in range(3):
            self.value += 1
            if self.value > 100:
                self.value = 1
            self.result.append(self.value)
            self.move += self.value
            self.counter += 1


# I'd seen this memoization mentioned in some of the solution threads related to
# searching the large data sets, if this was going to be solvable with brute force
# I needed to be speeded up somehow. Python 3.9 has @cache but not 2.
# https://www.geeksforgeeks.org/memoization-1d-2d-and-3d/
# https://docs.python.org/3.6/library/functools.html
# https://realpython.com/lru-cache-python/
from functools import lru_cache
@lru_cache(maxsize=None)
def score_search(position_1, score_1, position_2, score_2):
    win_count = [0,0]
    for roll_1 in itertools.product([1,2,3],repeat=3):
        for roll_2 in itertools.product([1,2,3],repeat=3):
            next_position_1 = position_1 + sum(roll_1)
            next_position_1 = ((next_position_1  - 1) % 10 + 1)  # Wrap at 10 
            next_score_1 = next_position_1 + score_1

            next_position_2 = position_2 + sum(roll_2)
            next_position_2 = ((next_position_2  - 1) % 10 + 1)
            next_score_2 = next_position_2 + score_2
            # See who won?
            if next_score_1 >= 21:
                win_count[0]+=1
                break  # Player 1 won, break from the inner loop.
            if next_score_2 >= 21:
                win_count[1]+=1
                continue
            # No one has won yet, begin recursion!
            next_win_count = score_search(next_position_1, next_score_1, next_position_2, next_score_2)
            # Add up all the recursive wins
            win_count = [a+b for a,b in zip(win_count, next_win_count)]
            #print(win_count)
    return win_count


def _day21(real_data=False):
    """
    Wasted time making a cool die class thinking maybe P2 would increase the players
    or change the number of dice, but no....
    """
    puzzle = [

    ]
    players = {1: {"position":4, "score":0},
               2: {"position":8, "score":0}}
    if real_data:
        players = {1: {"position":1, "score":0},
                   2: {"position":5, "score":0}}
    die = deterministic_die()
    stop = False
    while not stop:
        #print(players)
        for i in range(1,3):
            die.roll()
            #print(die.result)
            players[i]["position"] += die.move
            if players[i]["position"] > 10:
                players[i]["position"] =((players[i]["position"] - 1) % 10 + 1)
            players[i]["score"] += players[i]["position"]
            if players[i]["score"] >=1000:
                stop = True
                break
    #print(players)
    #print(die.counter)
    scores = []
    for k,v in players.items():
        scores.append(v["score"])
    print("Part 1")
    print(f" The player had {min(scores)} points and the die was rolled {die.counter} times")
    print(f" The answer is {min(scores)*die.counter}")
    # Part 2 using pretty much nothing of part 1.
    print("Part 2")
    s_time = time.time()
    # Using @lru_cache requires the arguments to be hashable, weird things happen when I pass it values directly from the dictionary.
    if real_data:
        total_wins = score_search(1, 0, 5, 0)
    else:
        #total_wins = score_search(copy.copy(players[1]["position"]), 0, copy.copy(players[2]["position"]), 0)
        total_wins = score_search(4, 0, 8, 0)
    #print(total_wins)
    print(" ",score_search.cache_info())
    print(f" The player that wins more won {max(total_wins)} times")
    print(f" That just took {time.time()-s_time:0.3f}s")


def _day22_p1(real_data=False):
    puzzle = [ "on x=10..12,y=10..12,z=10..12", "on x=11..13,y=11..13,z=11..13", "off x=9..11,y=9..11,z=9..11", "on x=10..10,y=10..10,z=10..10",    ]
    puzzle = [ "on x=-20..26,y=-36..17,z=-47..7", "on x=-20..33,y=-21..23,z=-26..28", "on x=-22..28,y=-29..23,z=-38..16", "on x=-46..7,y=-6..46,z=-50..-1", "on x=-49..1,y=-3..46,z=-24..28", "on x=2..47,y=-22..22,z=-23..27", "on x=-27..23,y=-28..26,z=-21..29", "on x=-39..5,y=-6..47,z=-3..44", "on x=-30..21,y=-8..43,z=-13..34", "on x=-22..26,y=-27..20,z=-29..19", "off x=-48..-32,y=26..41,z=-47..-37", "on x=-12..35,y=6..50,z=-50..-2", "off x=-48..-32,y=-32..-16,z=-15..-5", "on x=-18..26,y=-33..15,z=-7..46", "off x=-40..-22,y=-38..-28,z=23..41", "on x=-16..35,y=-41..10,z=-47..6", "off x=-32..-23,y=11..30,z=-14..3", "on x=-49..-5,y=-3..45,z=-29..18", "off x=18..30,y=-20..-8,z=-3..13", "on x=-41..9,y=-7..43,z=-33..15", "on x=-54112..-39298,y=-85059..-49293,z=-27449..7877", "on x=967..23432,y=45373..81175,z=27513..53682",]
    if real_data:
        puzzle = get_input(22, '\n', None)
    reactor = np.zeros((101,101,101))
    start_time = time.time()
    for line in puzzle:
        direction, coordinates = line.split(" ")
        coordinates = coordinates.split(",")
        cubiod = []
        for dim in coordinates:
            start = int(dim.split("=")[1].split("..")[0]) + 50  # Shift the cubes into positive space
            if start < 0 or start  >101:
                start = 0
            end = int(dim.split("=")[1].split("..")[1]) + 50 + 1
            if end < 0 or end > 101:
                end = 0
            cubiod.append((start,end))
        value = 1 if direction == "on" else 0
        reactor[cubiod[0][0]:cubiod[0][1],cubiod[1][0]:cubiod[1][1],cubiod[2][0]:cubiod[2][1]] = value
    print(f"The number of lit cubes is {np.count_nonzero(reactor)}")
    print(f"Calculation time: {time.time()-start_time:0.4f}")



class Borg:
    """
    A class to hold information about each cube.
    """
    def __init__(self, near_point, far_point, sign):
        self.near_point = near_point
        self.far_point = far_point
        self.sign = sign
        self.cubes = self.volume()

    def check_overlap(self, this_borg):
        if (this_borg.far_point[0] >= self.near_point[0] and this_borg.near_point[0] <= self.far_point[0]) and\
           (this_borg.far_point[1] >= self.near_point[1] and this_borg.near_point[1] <= self.far_point[1]) and\
           (this_borg.far_point[2] >= self.near_point[2] and this_borg.near_point[2] <= self.far_point[2]):
            return True
        else:
            return False

    def overlapping(self, this_borg):
        new_near_point = [0]*3
        new_far_point = [0]*3
        for d in range(3):
            new_near_point[d] = max(self.near_point[d],this_borg.near_point[d])
            new_far_point[d] = min(self.far_point[d],this_borg.far_point[d])
        return Borg(new_near_point, new_far_point, self.sign * -1)

    def volume(self):
        vol = 1
        for d in range(3):
            vol *= self.far_point[d] - self.near_point[d] + 1
        vol *= self.sign
        return vol


def _day22_p2(real_data=False):
    puzzle = ["on x=-5..47,y=-31..22,z=-19..33", "on x=-44..5,y=-27..21,z=-14..35", "on x=-49..-1,y=-11..42,z=-10..38", "on x=-20..34,y=-40..6,z=-44..1", "off x=26..39,y=40..50,z=-2..11",
              "on x=-41..5,y=-41..6,z=-36..8", "off x=-43..-33,y=-45..-28,z=7..25", "on x=-33..15,y=-32..19,z=-34..11", "off x=35..47,y=-46..-34,z=-11..5", "on x=-14..36,y=-6..44,z=-16..29",
              "on x=-57795..-6158,y=29564..72030,z=20435..90618", "on x=36731..105352,y=-21140..28532,z=16094..90401", "on x=30999..107136,y=-53464..15513,z=8553..71215", "on x=13528..83982,y=-99403..-27377,z=-24141..23996",
              "on x=-72682..-12347,y=18159..111354,z=7391..80950", "on x=-1060..80757,y=-65301..-20884,z=-103788..-16709", "on x=-83015..-9461,y=-72160..-8347,z=-81239..-26856", "on x=-52752..22273,y=-49450..9096,z=54442..119054",
              "on x=-29982..40483,y=-108474..-28371,z=-24328..38471", "on x=-4958..62750,y=40422..118853,z=-7672..65583", "on x=55694..108686,y=-43367..46958,z=-26781..48729", "on x=-98497..-18186,y=-63569..3412,z=1232..88485",
              "on x=-726..56291,y=-62629..13224,z=18033..85226", "on x=-110886..-34664,y=-81338..-8658,z=8914..63723", "on x=-55829..24974,y=-16897..54165,z=-121762..-28058", "on x=-65152..-11147,y=22489..91432,z=-58782..1780",
              "on x=-120100..-32970,y=-46592..27473,z=-11695..61039", "on x=-18631..37533,y=-124565..-50804,z=-35667..28308", "on x=-57817..18248,y=49321..117703,z=5745..55881", "on x=14781..98692,y=-1341..70827,z=15753..70151",
              "on x=-34419..55919,y=-19626..40991,z=39015..114138", "on x=-60785..11593,y=-56135..2999,z=-95368..-26915", "on x=-32178..58085,y=17647..101866,z=-91405..-8878", "on x=-53655..12091,y=50097..105568,z=-75335..-4862",
              "on x=-111166..-40997,y=-71714..2688,z=5609..50954", "on x=-16602..70118,y=-98693..-44401,z=5197..76897", "on x=16383..101554,y=4615..83635,z=-44907..18747", "off x=-95822..-15171,y=-19987..48940,z=10804..104439",
              "on x=-89813..-14614,y=16069..88491,z=-3297..45228", "on x=41075..99376,y=-20427..49978,z=-52012..13762", "on x=-21330..50085,y=-17944..62733,z=-112280..-30197", "on x=-16478..35915,y=36008..118594,z=-7885..47086",
              "off x=-98156..-27851,y=-49952..43171,z=-99005..-8456", "off x=2032..69770,y=-71013..4824,z=7471..94418", "on x=43670..120875,y=-42068..12382,z=-24787..38892", "off x=37514..111226,y=-45862..25743,z=-16714..54663",
              "off x=25699..97951,y=-30668..59918,z=-15349..69697", "off x=-44271..17935,y=-9516..60759,z=49131..112598", "on x=-61695..-5813,y=40978..94975,z=8655..80240", "off x=-101086..-9439,y=-7088..67543,z=33935..83858",
              "off x=18020..114017,y=-48931..32606,z=21474..89843", "off x=-77139..10506,y=-89994..-18797,z=-80..59318", "off x=8476..79288,y=-75520..11602,z=-96624..-24783", "on x=-47488..-1262,y=24338..100707,z=16292..72967",
              "off x=-84341..13987,y=2429..92914,z=-90671..-1318", "off x=-37810..49457,y=-71013..-7894,z=-105357..-13188", "off x=-27365..46395,y=31009..98017,z=15428..76570", "off x=-70369..-16548,y=22648..78696,z=-1892..86821",
              "on x=-53470..21291,y=-120233..-33476,z=-44150..38147", "off x=-93533..-4276,y=-16170..68771,z=-104985..-24507"]
    if real_data:
        puzzle = get_input(22, '\n', None)
    # Puzzle parsing and solution.
    borg_cubes = []  # list of all of the borg objects.
    on_count = 0
    start_time = time.time()
    for line in puzzle:
        # Puzzle data parsing, taking advantage of the fact the ranges are always
        # in increasing direction in the data set.
        on_off, coordinates = line.split(" ")
        on_off = 1 if on_off == "on" else -1
        coordinates = coordinates.split(",")
        near_corner = [0,0,0]
        far_corner = [0,0,0]

        # Split the data up into the near corner of the cuboid and the far one.
        for index, dim in enumerate(coordinates):
            near_corner[index] = int(dim.split("=")[1].split("..")[0])
            far_corner[index] = int(dim.split("=")[1].split("..")[1])
        # Create a new cube to track the changes (turn on/off)
        new_borg = Borg(near_corner, far_corner, on_off)

        # Iterate over all the existing cubes looking for ones that overlap this new sube.
        overlapping_cubes = []
        for i, borg in enumerate(borg_cubes):
            if borg.check_overlap(new_borg) is True:
                # Two cubes overlap which means you have one full cube and one cube with a missing part.
                # In the list of all cubes store 3 things, the original cube, the new cube, a cube used to 
                # "subtract" the overlap.
                overlapping_cubes.append(borg.overlapping(new_borg))
        borg_cubes.extend(overlapping_cubes)
        # Only append the new cube if it is one that turns things on.
        # The new off cube is effectively the "subtract" cubes from the above overlap check.
        if on_off == 1:
            borg_cubes.append(new_borg)
            on_count += 1
    # Done with the work. Walk through and sum up the cube volumes.
    cube_count = 0
    for b in borg_cubes:
        cube_count += b.cubes
    print(f"The initialization cube count is {cube_count}")
    print(f"Calculation time: {time.time()-start_time:0.4f}")


def go(day=21):
    switch = {
        1:  _day1,
        2:  _day2,
        3:  _day3,
        4:  _day4,
        5:  _day5,
        6:  _day6,
        7:  _day7,
        8:  _day8,
        9:  _day9,
        10: _day10,
        11: _day11,
        12: _day12,
        13: _day13,
        14: _day14,
        15: _day15,
        16: _day16,
        17: _day17,
        18: _day18,
        19: _day19,
        20: _day20,
        21: _day21,
    }
    return switch.get(day, "Invalid day")()
