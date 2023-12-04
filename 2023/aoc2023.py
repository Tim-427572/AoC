import re
# import sys
# import math
# import time
# import copy
# import curses # pip install windows-curses
import pickle
import socket
import pyglet
import hashlib
# import string
import requests
# import functools
# import itertools
# import statistics
# import collections
import numpy as np
from os import path


# Advent of Code
# Never did spend the time to work out how to get oAuth to work so this code expects you to
# manually copy over your session cookie value.
# Using a web browser inspect the cookies when logged into the Advent of Code website.
# Copy the value from the "session" cookie into a text file called "session.txt"

# Constants
_code_path = r'c:\AoC'
_offline = False
_year = 2023


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


def _pull_puzzle_input(day, seperator, cast):
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
def get_input(day, seperator, cast, override=False):
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


def day1(example=False):
    """
    So it begins!
    """
    day = 1
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


def day1_speed(example=0):
    day = 1
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
    calibration_doc = get_input(day, "\n", None, False)
    p1_calibration = p2_calibration = 0
    number_set = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
    number_forward_dict = {"o": [("one", "1")],
                           "e": [("eight", "8")],
                           "t": [("two", "2"), ("three","3")],
                           "f": [("four", "4"), ("five", "5")],
                           "s": [("six", "6"), ("seven", "7")],
                           "n": [("nine", "9")]}
    number_reverse_dict = {"e": [("one", "1"), ("three","3"), ("five", "5"), ("nine", "9")],
                           "o": [("two", "2")],
                           "r": [("four", "4")],
                           "x": [("six", "6")],
                           "n": [("seven", "7")],
                           "t": [("eight", "8")]}
    for new_value in calibration_doc:
        first_digit = first_word = first_thing = None
        forward = new_value
        while forward:
            if first_digit is None and forward[0] in number_set:
                first_digit = forward[0]
                first_thing = first_digit if first_thing is None else first_thing
            if first_word is None and forward[0] in number_forward_dict.keys():
                for number_word, value in number_forward_dict[forward[0]]:
                    if forward.startswith(number_word):
                        first_word = value
                        break
                first_thing = first_word if first_thing is None else first_thing
            if first_digit is not None and first_word is not None:
                break
            forward = forward[1:]
        last_digit = last_word = last_thing = None
        backwards = new_value
        while backwards:
            if last_digit is None and backwards[-1] in number_set:
                last_digit = backwards[-1]
                last_thing = last_digit if last_thing is None else last_thing
            if last_word is None and backwards[-1] in number_reverse_dict.keys():
                for number_word, value in number_reverse_dict[backwards[-1]]:
                    if backwards.endswith(number_word):
                        last_word = value
                        break
                last_thing = last_word if last_thing is None else last_thing
            #print(backwards, last_digit, last_word)
            if last_digit is not None and last_word is not None:
                break
            backwards = backwards[:-1]
        # Only need this conditional to avoid an error when running example #2.
        # p1_calibration += int(f"{first_digit}{last_digit}") if first_digit is not None and last_digit is not None else p1_calibration
        p1_calibration += int(f"{first_digit}{last_digit}")
        p2_calibration += int(f"{first_thing}{last_thing}")
    print(f"Part 1 the sum of calibration values is {p1_calibration}")
    print(f"Part 2 the sum of calibration values is {p2_calibration}")


def day2(example=False, reload=False):
    day = 2 if not example else ("Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green\n"
                                 "Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue\n"
                                 "Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red\n"
                                 "Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red\n"
                                 "Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green\n")
    puzzle = get_input(day, "\n", None, reload)
    p1_cubes = {"red":12, "green":13, "blue":14} 
    p1_possible_sum = p2_power_sum = 0
    for game in puzzle:
        game_id, game_info = game.split(":")
        game_id = game_id.split(" ")[1]
        possible_game = True
        min_cubes = {"red":0, "green":0, "blue":0} 
        for subset in game_info.split(";"):
            for cubes in subset.split(","):
                number, color = cubes.strip().split(" ")
                number = int(number)
                possible_game = False if p1_cubes[color] < number else possible_game
                min_cubes[color] = number if min_cubes[color] < number else min_cubes[color]
        if possible_game:
            p1_possible_sum += int(game_id)
        p2_power_sum += np.prod(list(min_cubes.values()))
    print(f"Part 1 sum of possible game IDs is {p1_possible_sum}")
    print(f"Part 2 sum of game cube power is {p2_power_sum}")


def day3(example=False, reload=False):
    from numpy.lib import stride_tricks
    day = 3 if not example else ("467..114..\n"
                                 "...*......\n"
                                 "..35..633.\n"
                                 "......#...\n"
                                 "617*......\n"
                                 ".....+.58.\n"
                                 "..592.....\n"
                                 "......755.\n"
                                 "...$.*....\n"
                                 ".664.598..\n")
    schematic = get_input(day, "\n", None)
    schematic_array = np.array(list(map(list, schematic)))
    schematic_array = np.pad(schematic_array, 1, 'constant', constant_values = ".")
    shape = (schematic_array.shape[0] - 2, schematic_array.shape[1] - 4, 3, 5)
    stride = schematic_array.strides * 2
    patches = stride_tricks.as_strided(schematic_array, shape=shape, strides=stride)
    #return patches
    engine_part_sum = 0
    for row in patches:
        for patch in row:
            if np.char.isdigit(patch[1,3]) and not np.char.isdigit(patch[1,4]):  # Found the left edge of a number
                print(patch)
                if not patch[1, 2].isdigit():  # one digit number
                    possible_num = int("".join(patch[1,3:4]))
                    patch = patch[:,2:]
                elif not patch[1, 1].isdigit():  # two digit number
                    possible_num = int("".join(patch[1,2:4]))
                    patch = patch[:,1:]
                else:
                    possible_num = int("".join(patch[1,1:4]))
                #np.place(patch, np.char.isdigit(patch), ".")
                non_digit = patch[~np.char.isdigit(patch)]
                if np.count_nonzero( non_digit=="." ) != non_digit.size:
                    #print(possible_num)
                    engine_part_sum += int(possible_num)
    print(engine_part_sum)
    #return patches


def day3_p2(example=False, reload=False):
    from numpy.lib import stride_tricks
    day = 3 if not example else ("467..114..\n"
                                 "...*......\n"
                                 "..35..633.\n"
                                 "......#...\n"
                                 "617*......\n"
                                 ".....+.58.\n"
                                 "..592.....\n"
                                 "......755.\n"
                                 "...$.*....\n"
                                 ".664.598..\n")
    schematic = get_input(day, "\n", None)
    schematic_array = np.array(list(map(list, schematic)))
    schematic_array = np.pad(schematic_array, 1, 'constant', constant_values = ".")
    shape = (schematic_array.shape[0] - 2, schematic_array.shape[1] - 6, 3, 7)
    stride = schematic_array.strides * 2
    patches = stride_tricks.as_strided(schematic_array, shape=shape, strides=stride)
    gear_ratio_sum = 0
    for thing in patches:
        for patch in thing:
            if patch[1,3] == "*":
                test = patch[:,2:5]
                numbers = 0
                for row in test:
                    digit_count = np.count_nonzero(np.char.isdigit(row))
                    if digit_count:
                        if digit_count > 1 and not np.char.isdigit(row[1]):
                            numbers += 2
                        else:
                            numbers += 1
                if numbers == 2:
                    #print(patch)
                    gears = []
                    for row in patch:
                        if np.char.isdigit(row[3]):
                            num = row[3]
                            for l in row[4:6]:
                                if l.isdigit():
                                    num += l
                                else:
                                    break
                            for l in np.flip(row[1:3]):
                                if l.isdigit():
                                    num = l+num
                                else:
                                    break
                            gears.append(int(num))
                        if not np.char.isdigit(row[3]):
                            if np.char.isdigit(row[2]):
                                num = ""
                                for l in np.flip(row[:3]):
                                    if l.isdigit():
                                        num = l + num
                                    else:
                                        break
                                gears.append(int(num))
                            if np.char.isdigit(row[4]):
                                num = ""
                                for l in row[4:]:
                                    if l.isdigit():
                                        num += l
                                    else:
                                        break
                                gears.append(int(num))
                    #print(gears)
                    gear_ratio_sum += np.prod(gears)
    print(gear_ratio_sum)
    #return patches


def day3(example=False, reload=False):
    day = 3 if not example else ("467..114..\n"
                                 "...*......\n"
                                 "..35..633.\n"
                                 "......#...\n"
                                 "617*......\n"
                                 ".....+.58.\n"
                                 "..592.....\n"
                                 "......755.\n"
                                 "...$.*....\n"
                                 ".664.598..\n")
    schematic = get_input(day, "\n", None)
    schematic_array = np.array(list(map(list, schematic)))
    schematic_array = np.pad(schematic_array, 1, 'constant', constant_values = ".")
    number_positions = {}
    number_hashes = {}
    num_regex = re.compile("[0-9]\w+")
    for row_idx, row in enumerate(schematic): 
        numbers = num_regex.finditer(row)


    

def day4(example=False, reload=False):
    day = 4 if not example else """Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11
"""
    puzzle = get_input(day, "\n", None, reload)
    cards = {}
    score = 0
    for card in puzzle:
        card_number, values = card.split(":")
        winning, have = values.split("|")
        card_number = int(card_number.split()[1])
        win_set = set(map(int, winning.split()))
        have_set = set(map(int, have.split()))
        matches = len(win_set.intersection(have_set))
        if matches:
            score += 2**(matches-1)
        cards[card_number] = matches
    print(f"Part 1 the cards are worth {score}")
    orig_card_list = list(cards.keys())
    final_list = []
    while orig_card_list:
        card = orig_card_list.pop()
        final_list.append(card)
        if cards[card]:
           orig_card_list += list(range(card + 1, (card + cards[card] + 1)))
    print(f"Part 2 the total number of scratchcards is {len(final_list)}")

class Viz(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_draw(self):
        self.clear()
        label = pyglet.text.Label("Hello, world",font_name='Times New Roman',font_size=36, x=self.width//2, y=self.height//2, anchor_x='center',anchor_y='center')
        label.draw()

def v():
    #label = pyglet.text.Label("Hello, world",font_name='Times New Roman',font_size=36, anchor_x='center',anchor_y='center')
    _ = Viz(512, 512, "Test",resizable=False)
    pyglet.app.run()

def go(day=1):
    try:
        return eval("day{}".format(day))
    except Exception as e:
        print(e)

