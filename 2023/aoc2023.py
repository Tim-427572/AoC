import hashlib
# import sys
# import math
# import time
# import copy
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
import numpy as np
import pyglet
# import string
import requests
import functools

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
    """
    day_input = get_input(day, seperator, cast, override)
    if splitter is None:
        return np.array(day_input, dtype=dtype)
    else:
        temp = []
        for r in day_input:
            temp.append(splitter(r))
        return np.array(temp, dtype=dtype)


def print_np(array):
    if array.dtype == np.dtype("<U1"):
        for row in array:
            print("".join(row))
    else:
        for row in array:
            print(np.array2string(row, separator="")[1:-1])



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


class Point_Object:
    """
    A point object to use with 2D arrays where y/row is the first index and x/column is the second.
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


class Coordinate(tuple):
    def __add__(self, other):
        return Coordinate(x + y for x, y in zip(self, other))
    def __setitem__(self, key, value):
        l = list(self)
        l[key] = value
        print(l)
        return Coordinate(tuple(l))


move_dict = {"u":(-1, 0), "n":(-1, 0), "up":(-1, 0), "north":(-1, 0),
             "d":(1, 0), "s":(1, 0), "down":(1,0), "south":(1, 0),
             "r":(0, 1), "e":(0, 1), "right":(0, 1), "east":(0, 1),
             "l":(0, -1), "w":(0, -1), "left":(0, -1), "west":(0, -1),
             "ur":(-1, 1), "ne":(-1, 1), "up-right":(-1, 1), "north-east":(-1, 1),
             "dr":(1, 1), "se":(1, 1), "down-right":(1, 1), "south-east":(1, 1),
             "ul":(-1, -1), "nw":(-1, -1), "up-left":(-1, -1), "north-west":(-1, -1),
             "dl":(1, -1), "sw":(1, -1), "down-left":(1, -1), "south-west":(1, -1)}


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


def day2(example=False, reload=False):
    if example:
        day = ("Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green\n"
               "Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue\n"
               "Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red\n"
               "Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red\n"
               "Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))

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
    if example:
        day = ("467..114..\n"
               "...*......\n"
               "..35..633.\n"
               "......#...\n"
               "617*......\n"
               ".....+.58.\n"
               "..592.....\n"
               "......755.\n"
               "...$.*....\n"
               ".664.598..\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    schematic = get_input(day, "\n", lambda x: "." + x + ".", None)
    schematic = ["."*len(schematic[0])] + list(schematic) + ["."*len(schematic[0])] 
    schematic_array = np.array(list(map(list, schematic)))
    number_positions = number_hashes = {}
    p1_sum = p2_sum = 0
    num_regex = re.compile("[0-9]+")
    for row_idx, row in enumerate(schematic): 
        numbers = num_regex.finditer(row)
        for number in numbers:
            num_hash = random.getrandbits(128)
            number_hashes[num_hash] = int(number.group())
            for i in range(*number.span()):
                number_positions[(row_idx, i)] = num_hash
    p1_adj_numbers = set()
    for symbol_r, symbol_c in zip(*np.where(~np.char.equal(schematic_array, ".") & ~np.char.isdigit(schematic_array))):
        p2_adj_gears = set()
        for row in range(symbol_r - 1, symbol_r + 2):
            for col in range(symbol_c - 1, symbol_c + 2):
                if (row, col) in number_positions.keys():
                    p1_adj_numbers.add(number_positions[(row, col)])
                    p2_adj_gears.add(number_positions[(row, col)])
        if len(p2_adj_gears) == 2:
            p2_sum += np.prod(list(map(lambda x: number_hashes[x], p2_adj_gears)))
    p1_sum = sum(list(map(lambda x: number_hashes[x], p1_adj_numbers)))

    print(f"Part 1 the sum of engine part numbers is {p1_sum}")
    print(f"Part 2 the sum of engine gears ratio {p2_sum}")


def day4(example=False, reload=False):
    if example:
        day = ("Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53\n"
               "Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19\n"
               "Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1\n"
               "Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83\n"
               "Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36\n"
               "Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
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


def _day5_fwd(seed, ranges):
    for dest, src, size in ranges:
        if src <= seed < src + size:
            return seed + dest - src
    return seed

def _day5_range(ranges, this_map):
    result = []
    for dest, src, size in this_map:
        src_end = src + size
        new_ranges = []
        # Look for ranges that overlap this mapping section
        # If it does overlap, remap the range based on the rules.
        while ranges:
            start, end = ranges.pop()
            left = (start, min(end, src))
            inside = (max(start, src), min(src_end, end))
            right = (max(src_end, start), end)
            if left[1] > left[0]:
                new_ranges.append(left)
            if inside[1] > inside[0]:
                result.append((inside[0] - src + dest, inside[1] - src + dest))
            if right[1] > right[0]:
                new_ranges.append(right)
        ranges = new_ranges
    return result + ranges


def day5(example=False, reload=False):
    if example:
        day = ("seeds: 79 14 55 13\n"
               "\n"
               "seed-to-soil map:\n"
               "50 98 2\n"
               "52 50 48\n"
               "\n"
               "soil-to-fertilizer map:\n"
               "0 15 37\n"
               "37 52 2\n"
               "39 0 15\n"
               "\n"
               "fertilizer-to-water map:\n"
               "49 53 8\n"
               "0 11 42\n"
               "42 0 7\n"
               "57 7 4\n"
               "\n"
               "water-to-light map:\n"
               "88 18 7\n"
               "18 25 70\n"
               "\n"
               "light-to-temperature map:\n"
               "45 77 23\n"
               "81 45 19\n"
               "68 64 13\n"
               "\n"
               "temperature-to-humidity map:\n"
               "0 69 1\n"
               "1 0 69\n"
               "\n"
               "humidity-to-location map:\n"
               "60 56 37\n"
               "56 93 4\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    maps = []
    ranges = []
    i = 0
    while i < len(puzzle):
        line = puzzle[i]
        if "seeds" in line:
            seeds = list(map(int, (line.split(":")[1].split())))
            i += 1
        elif line == "":
            if ranges:
                ranges.sort()
                maps.append(ranges.copy())
                ranges = []
            i+=1
        elif ":" in line: # one of the map names.
            i+=1
        else:
            dest, src, size = map(int, line.split())
            ranges.append((dest, src, size))
            i+=1
    if ranges:  # Final range at the end of the input.
        ranges.sort()
        maps.append(ranges.copy())

    p1_min = None
    for seed in seeds:
        for ranges in maps:
            seed = _day5_fwd(seed, ranges)
        p1_min = seed if( p1_min is None or seed < p1_min) else p1_min
    print(f"Part 1 the minimum location is {p1_min}")

    p2_min = None
    seed_ranges = list(zip(*[iter(seeds)]*2))
    # Walk the seed ranges
    for start, size in seed_ranges:
        ranges = [(start, start+size)]
        for this_map in maps:
            ranges = _day5_range(ranges, this_map)
        range_min = min(ranges)[0]
        p2_min = range_min if p2_min is None or range_min < p2_min else p2_min
    print(f"Part 2 the minimum location is {p2_min}")


def day6(example=False, reload=False):
    if example:
        day = ("Time:      7 15 30\n"
               "Distance:  9 40 200\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    p1_answer = p2_answer = 1
    times = list(map(int, puzzle[0].split(":")[1].split()))
    distances = list(map(int, puzzle[1].split(":")[1].split()))
    times.append(int(puzzle[0].split(":")[1].replace(" ", "")))
    distances.append(int(puzzle[1].split(":")[1].replace(" ","")))
    for race_time, distance in zip(times, distances):
        ways = 0
        for hold in range(race_time):
            if hold * (race_time - hold) > distance:
                ways += 1
            elif ways != 0:
                break
        if race_time == times[-1]:
            p2_answer *= ways
        else:
            p1_answer *= ways
    print(f"The part 1 answer is {p1_answer}")
    print(f"The part 2 answer is {p2_answer}")


def _day7_check_type(hand):
    hand_decode = {(5,1):"five of a kind",
                   (4,1):"four of a kind",
                   (3,2):"full house",
                   (3,1):"three of a kind",
                   (2,2):"two pair",
                   (2,1):"one pair",
                   (1,1):"high card"}
    counts = collections.Counter(hand + "!").most_common(2) # Pad with ! to handle the five of a kind case.
    return hand_decode[list(zip(*counts))[1]]


def _day7_hand_sort(left, right):
    """
    Custom sorting function for the strength of the hands
    going to use 1 for the J when scoring part instead of having two of these functions.
    """
    card_value = {"1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
    for left_char, right_char in zip(left, right):
        if card_value[left_char] > card_value[right_char]:
            return 1
        if card_value[left_char] < card_value[right_char]:
            return -1
    return 0

def day7(example=False, reload=False):
    if example:
        day = ("32T3K 765\n"
               "T55J5 684\n"
               "KK677 28\n"
               "KTJJT 220\n"
               "QQQJA 483\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    type_order = ["high card", "one pair", "two pair", "three of a kind", "full house", "four of a kind", "five of a kind"]
    p1_answer = p2_answer = 0
    p1_hands_of_each_type = collections.defaultdict(list)
    p2_hands_of_each_type = collections.defaultdict(list)
    hand_bid = {}
    for line in puzzle:
        hand = line.split()[0]
        score = int(line.split()[1])
        hand_bid[hand] = score
        p1_hands_of_each_type[_day7_check_type(hand)].append(hand)
        # Replace the J wild cards to match the most common other card.
        if "J" in hand and hand.count("J") < 5:
            new_hand = hand.replace("J", collections.Counter(hand.replace("J", "")).most_common(1)[0][0])
        else:
            new_hand = hand
        p2_hands_of_each_type[_day7_check_type(new_hand)].append(hand.replace("J", "1"))  # Replace J with 1 so I don't need two sorting functions.
    p1_strengths = []
    p2_strengths = []
    for hand_type in type_order:
        p1_strengths += sorted(p1_hands_of_each_type[hand_type], key=functools.cmp_to_key(_day7_hand_sort))
        p2_strengths += sorted(p2_hands_of_each_type[hand_type], key=functools.cmp_to_key(_day7_hand_sort))

    # Ok we have the hands in order, time to score them.
    rank = 1
    for p1_hand, p2_hand in zip(p1_strengths, p2_strengths):
        p1_answer += rank * hand_bid[p1_hand]
        p2_answer += rank * hand_bid[p2_hand.replace("1", "J")]  # Put the J back so we can look up the score.
        rank += 1
    
    print(f"The part 1 total winnings is {p1_answer}")
    print(f"The part 2 total winnings is {p2_answer}")


def day8(example=False, reload=False):
    if example:
        day = ("LR\n"
                "\n"
                "11A = (11B, XXX)\n"
                "11B = (XXX, 11Z)\n"
                "11Z = (11B, XXX)\n"
                "22A = (22B, XXX)\n"
                "22B = (22C, 22C)\n"
                "22C = (22Z, 22Z)\n"
                "22Z = (22B, 22B)\n"
                "XXX = (XXX, XXX)\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    instructions = list(map(int, puzzle[0].replace("R","1").replace("L","0")))
    instructions = itertools.cycle(instructions)
    puzzle = puzzle[2:]
    p1_answer = 0
    nodes = {}
    a_elements = []
    step_list = []
    for line in puzzle:
        this_element, left, right = re.findall("[0-9A-Z]+", line)
        nodes[this_element] = (left, right)
        if this_element[-1] == "A":
            a_elements.append(this_element)
    for element in a_elements:
        steps = 0
        while True:
            element = nodes[element][next(instructions)]
            steps += 1
            p1_answer = steps if element == "ZZZ" else p1_answer
            if element[-1] == "Z":
                step_list.append(steps)
                break
    print(f"The totals steps for part 1 is {p1_answer}")
    print(f"The part 2 answer is {np.lcm.reduce(step_list, dtype='int64')}")



def day9(example=False, reload=False):
    if example:
        day = ("0 3 6 9 12 15\n"
               "1 3 6 10 15 21\n"
               "10 13 16 21 30 45\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    puzzle = get_input(day, "\n", None, reload)
    p1_answer = []
    p2_answer = []
    value_histories = []
    for line in puzzle:
        value_histories.append([list(map(int, line.split()))])
    # Fill in each value hist
    for value_history in value_histories:
        while True:
            value_history.append(np.diff(value_history[-1]))
            if all(x ==0 for x in value_history[-1]):
                break
    for value_history in value_histories:
        p1 = p2 = 0
        for i in range(len(value_history) - 1, -1, -1):
            p1=value_history[i][0] + p1
            p2=value_history[i][0] - p2
        p1_answer.append(p1)
        p2_answer.append(p2)
    print(f"Part 1 sum is {sum(p1_answer)}")
    print(f"Part 2 sum is {sum(p2_answer)}")



def d10_bfs(field, loop, coordinate, dist):  # BFS search? realy only takes the two paths around the loop.
    global move_dict
    queue = [(coordinate, dist)]
    """
    """
    while queue:
        this_node, this_distance = queue.pop(0)
        loop[this_node] = this_distance
        for direction, here, there in [("up","|SJL","S|7F"),
                                       ("down","S|7F","S|LJ"),
                                       ("left","S-J7","S-FL"),
                                       ("right","S-FL","S-J7")]:
            neighbor = ths_node + move_dict[direction]
            #print(this_node, neighbor)
            if (neighbor not in loop and
                field[this_node] in here and
                field[neighbor] in there):
                queue.append((neighbor, this_distance + 1))


def day10(example=False, reload=False):
    if example:
        day = """..........
.S------7.
.|F----7|.
.||....||.
.||....||.
.|L-7F-J|.
.|..||..|.
.L--JL--J.
..........
"""        
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    field = get_np_input(day, "\n", splitter=list, dtype=str, override=reload)
    field = np.pad(field, 1, mode="constant", constant_values=".")
    #print_np(field)
    start = Coordinate(np.argwhere(field == "S")[0])
    print(start)
    loop = {}
    d10_bfs(field, loop, start, 0)
    print(loop)
    return field



def day11(universe_expansion = 2, example=False, reload=False,):
    if example:
        day = ("...#......\n"
               ".......#..\n"
               "#.........\n"
               "..........\n"
               "......#...\n"
               ".#........\n"
               ".........#\n"
               "..........\n"
               ".......#..\n"
               "#...#.....\n")
    else:
        day = int(inspect.currentframe().f_code.co_name.split("_")[0].strip("day"))
    universe = get_np_input(day, "\n", splitter=list, dtype=str, override=reload)
    empty_rows = np.where(~(universe == "#").any(axis=1))[0].tolist()
    empty_columns = np.where(~(universe == "#").any(axis=0))[0].tolist()
    galaxies = np.argwhere(universe == "#").tolist()
    universe_expansion -= 1  # Accound for the one existing empty line in the universe map.
    answer = 0
    for p1, p2 in itertools.combinations(galaxies, 2):
        row_range = range(*sorted([p1[0], p2[0]]))
        row_delta = len(row_range)
        row_delta += sum([universe_expansion for row in empty_rows if row in row_range])
        col_range = range(*sorted([p1[1], p2[1]]))
        col_delta = len(col_range)
        row_delta += sum([universe_expansion for col in empty_columns if col in col_range])
        # print(pair, (row_delta + col_delta))  # Was helping Max debug.
        answer += (row_delta + col_delta)
    print(f"The sum of the distances between galaxies is {answer}")

