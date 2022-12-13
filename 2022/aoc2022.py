import re
import sys
import math
import time
import copy
import pickle
import socket
import string
import requests
import itertools
import statistics
import numpy as np
from os import path
from functools import lru_cache
from collections import defaultdict

# Advent of Code
# Never did spend the time to work out how to get oAuth to work so this code expects you to
# manually copy over your session cookie value.
# Using a web browser inspect the cookies when logged into the Advent of Code website.
# Copy the value from the "session" cookie into a text file called "session.txt"

# Constants
_code_path = r'c:\AoC'
_offline = False
_year = 2022


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
    elif type(day) is str:  # An example string
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
        if type(day) is int:  # only save the full puzzle data to the pickle file.
            puzzle_dict[day] = puzzle_input
            pickle.dump(puzzle_dict, open(_code_path + r'\{}\input.p'.format(_year), 'wb'))
    return puzzle_input


def _day1():
    """
    How many calories does an elf carry
    """
    day = "1000\n2000\n3000\n\n4000\n\n5000\n6000\n\n7000\n8000\n9000\n\n10000\n"
    day = 1
    puzzle = get_input(day, '\n', None)
    elves = []
    cal = 0
    for val in puzzle:
        if val == "":
            elves.append(cal)
            cal=0
        else:
            cal += int(val)
    elves.append(cal)  # Last elf (might not be a \n after the last calorie data point)
    elves.sort(reverse=True)
    print(f"Single most calories {elves[0]}")
    print(f"Total of the top three elves {sum(elves[:3])}")


def _day1_eval():
    day = "1000\n2000\n3000\n\n4000\n\n5000\n6000\n\n7000\n8000\n9000\n\n10000\n"
    day = 1
    puzzle = list(get_input(day, '\n\n', None, True))
    puzzle[-1]=puzzle[-1].strip('\n')  # Formatting fix
    elves = [eval(x.replace("\n",'+')) for x in puzzle]
    elves.sort(reverse=True)
    print(f"Single most calories {elves[0]}")
    print(f"Total of the top three elves {sum(elves[:3])}")


def _day2_orig():
    """
    Rock Paper Scissors
    """
    day = "A Y\nB X\nC Z"
    day = 2
    puzzle = get_input(day, '\n', None)
    score = {"X":1, "Y":2, "Z":3}
    win =  {"A":"Y", "B":"Z", "C":"X"}  # What to have to win
    lose = {"A":"Z", "B":"X", "C":"Y"}  # What to pick to lose
    draw = {"A":"X", "B":"Y", "C":"Z"}  # Lookup for a draw
    total_score = 0
    for move in puzzle:
        elf, me = move.split(" ")
        if win[elf] == me:
            total_score += 6
        elif draw[elf] == me:
            total_score += 3
        else:
            total_score += 0
        total_score += score[me]
    print(f"Part 1 total score {total_score}")
    total_score = 0
    for move in puzzle:
        elf, result = move.split(" ")
        if result == "Z":  # Win
            me = win[elf]
            total_score += 6
        elif result == "Y":  # Draw
            me = draw[elf]
            total_score += 3
        else:
            me = lose[elf]
        total_score += score[me]
    print(f"Part 2 total score {total_score}")


def _day2():
    """
    Rock, Paper, Scissors with Elves!
    """
    day = "A Y\nB X\nC Z"
    day = 2
    puzzle = get_input(day, '\n', None)
    p1_dict=  {"B X":1, "C Y":2, "A Z":3, "A X":4, "B Y":5, "C Z":6, "C X":7, "A Y":8, "B Z":9}
    p2_dict = {"B X":1, "C X":2, "A X":3, "A Y":4, "B Y":5, "C Y":6, "C Z":7, "A Z":8, "B Z":9}
    print(f"Part 1 total score {sum([p1_dict[move] for move in puzzle])}")
    print(f"Part 2 total score {sum([p2_dict[move] for move in puzzle])}")


def _day3():
    """
    What's in your rucksack?!?
    """
    day = "vJrwpWtwJgWrhcsFMMfFFhFp\njqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL\nPmmdzqPrVvPwwTWBwg\nwMqvLMZHhHMvwLHjbvcjnnSBnvTQFn\nttgJtRGJQctTZtZT\nCrZsJsPPZsGzwwsLwLmpwMDw"
    day = 3
    puzzle = get_input(day, '\n', None)
    # Make a string of all lower and upper case letter to use to get the score for each letter.
    scoring_guide = " " + string.ascii_lowercase + string.ascii_uppercase
    score = 0
    for rucksack in puzzle:
        half = len(rucksack) // 2
        compartments = set.intersection(set(rucksack[:half]), set(rucksack[half:]))
        if len(compartments) != 1:
            raise Exception("Part 1; What? {compartments}")
        score += scoring_guide.find(compartments.pop())
    print(f"Part 1 priorities sum is {score}")
    score = 0
    for one, two, three in zip(*[iter(puzzle)]*3):
        group = set.intersection(set(one), set(two), set(three))
        if len(group) != 1:
            raise Exception(f"Part 2; What? {group}")
        score += scoring_guide.find(group.pop())
    print(f"Part 2 priorities sum is {score}")


def _day4():
    """
    Jungle clear-cutting elves.
    """
    day ="2-4,6-8\n2-3,4-5\n5-7,7-9\n2-8,3-7\n6-6,4-6\n2-6,4-8"
    day = 4
    puzzle = get_input(day, '\n', None)
    p1_score = p2_score = 0
    for pair in puzzle:
        first, second = pair.split(",")
        first_start, first_end = map(int, first.split('-'))
        second_start, second_end = map(int, second.split('-'))
        first_set = set(range(first_start, first_end + 1))
        second_set = set(range(second_start, second_end + 1))
        if first_set.issubset(second_set) or second_set.issubset(first_set):
            p1_score += 1
        if first_set.intersection(second_set):
            p2_score += 1
    print(f"Part 1 {p1_score} assignment pairs fully contian the other")
    print(f"Part 2 {p2_score} assignments overlap")


def _day5():
    """
    Well at least it wasn't one of those three tower colored ring puzzles.
    """
    day = "    [D]    \n[N] [C]    \n[Z] [M] [P]\n 1   2   3 \n\nmove 1 from 2 to 1\nmove 3 from 1 to 3\nmove 2 from 2 to 1\nmove 1 from 1 to 2"
    day = 5
    puzzle = get_input(day, "\n", None)
    # Get the number of stacks to make processing the table easier.
    stack_numbers = []
    for line in puzzle:
        if line.startswith(" 1 "):
            stacks = line.strip().split(" ")
            for char in stacks:
                if char:
                    stack_numbers.append(char)
            break

    p1_stacks_dict = {}
    for line in puzzle:
        if not line:
            continue
        elif line.startswith(" 1 "):
            # This is the line with the stack numbers, done loading now. Create a copy for part 2.
            p2_stacks_dict = copy.deepcopy(p1_stacks_dict)
        elif "move" in line:  # This is a stack movement line
            temp = []
            directions = line.split(" ")
            number_of_crates = -1 * int(directions[1])
            # Part 1 crates are reversed because they get picked 1 by 1.
            p1_stacks_dict[directions[5]] += reversed(p1_stacks_dict[directions[3]][number_of_crates:])
            del p1_stacks_dict[directions[3]][number_of_crates:]
            # Part 2 crates order is not reversed.
            p2_stacks_dict[directions[5]] += p2_stacks_dict[directions[3]][number_of_crates:]
            del p2_stacks_dict[directions[3]][number_of_crates:]            
        else:  # Note: index 0 is the bottom of the stack of crates.
            crates = [" "] + list(zip(*[iter(line + " ")] * 4))  # Pad because the stacks are 1's based numbering.
            for stack in stack_numbers:
                if crates[int(stack)][1] != " ":
                    p1_stacks_dict.setdefault(stack, [])
                    p1_stacks_dict[stack].insert(0, crates[int(stack)][1])

    p1_answer = p2_answer = ""
    for stack in stack_numbers:
        p1_answer += p1_stacks_dict[stack][-1]
        p2_answer += p2_stacks_dict[stack][-1]
    print(f"Part 1 top of stacks is {p1_answer}")
    print(f"Part 2 top of stacks is {p2_answer}")


def _start_of_packet(message, length):
    for i in range(length, len(message)):
        start = i - length
        if len(set(message[start:i])) == length:
            return i


def _day6():
    """
    Communication system.
    """
    # day = """mjqjpqmgbljsphdztnvjfqwrcgsmlb"""
    # day = "bvwbjplbgvbhsrlpgdmjqwftvncz"
    # day = "zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw"
    day = 6
    puzzle = get_input(day, "", None, True)[0]
    for length in [4, 14]:
        for i in range(len(puzzle) + 1 - length):
            if len(set(puzzle[i:i+length])) == length:
                print(f"Unique {length} character pattern {puzzle[i:i+length]} located at {i + length} ")
                break


def get_size(folder, drive_dict, folders_dict):
    """
    Recursive function to calculate the size of the folder and sub-folders.
    """
    total_size = 0
    file_dict = drive_dict[folder].get("files", {})
    for file, size in file_dict.items():
        total_size += size
    sub_folders = drive_dict[folder].get("sub-folders", set())
    for sub_folder in sub_folders:
        total_size += get_size(sub_folder, drive_dict, folders_dict)
    folders_dict.setdefault(folder, total_size)
    return total_size


def _day7():
    """
    Duplicate file names, what a pain...
    """
    day = 7
    puzzle = get_input(day, '\n', None, True)
    current_dir = "/."
    drive_dict = {"/.": {}}

    # Parsing
    ls = False
    for line in puzzle:
        if line.startswith("$ cd"):
            if ls:  # ls command is complete.
                drive_dict[current_dir]["files"] = folder
                folder = {}
                ls = False
            name = line.split(" ")[-1]
            if name == "..":  # Up to parent.
                current_dir = drive_dict[current_dir][".."]
            elif name == "/":  # Change to the root folder.
                current_dir = "/."
            else:
                new_name = current_dir + "/" + name  # Full file path (to handle duplicate folder names)
                drive_dict.setdefault(new_name, {})
                drive_dict[new_name][".."] = current_dir
                current_dir = new_name
        elif line.startswith("$ ls"):
            folder = {}
            ls = True
        elif line.startswith("dir"):
            folder_name = line.split(" ")[-1]
            drive_dict[current_dir].setdefault("sub-folders",set())
            drive_dict[current_dir]["sub-folders"].add(current_dir + "/" + folder_name)
        else:
            print(line)
            size, f_name = line.split(" ")
            folder[f_name] = int(size)
    if folder != {}:  # Just in case ls was the last command.
        drive_dict[current_dir]["files"] = folder

    # The actual solution.
    folders_dict = {}
    get_size('/.', drive_dict, folders_dict)
    sizes = np.array(list(folders_dict.values()))
    need_to_free = 30000000 - (70000000 - folders_dict["/."])
    smallest_to_delete = sizes[sizes > need_to_free].min()
    sizes[sizes > 100000] = 0
    print(sizes)
    print(f"Part 1 the total size is {sum(sizes)}")
    print(f"Part 2 Need to free {need_to_free} and the smallest folder to delete is {smallest_to_delete}")


def _day8(example=True):
    """
    """
    day = "30373\n25512\n65332\n33549\n35390"
    day = 8
    puzzle = get_input(day, '\n', list, True)
    n=np.array(puzzle, int)
    v_tree = set()
    max_score = 0
    for i in range(len(n)):
        for j in range(len(n[i])):
            # Boundry checks, tree at the edge is visible for part 1 and not interesting for part 2.
            if i == 0 or i == len(n[i])-1:
                v_tree.add((i,j))
                continue
            if j == 0 or j == len(n[:,i])-1:
                v_tree.add((i,j))
                continue
            tree_count = 0
            score = 1

            # left
            l = np.flip(n[i][:j])
            # part 1
            if len(l[l>=n[i][j]]) == 0:
                v_tree.add((i,j))
            # part 2
            for x in l:
                if x < n[i][j]:
                    tree_count+=1
                elif x >= n[i][j]:
                    tree_count+=1
                    break
            score *= tree_count
            tree_count = 0
                
            # right
            l = n[i][j+1:]
            if len(l[l>=n[i][j]]) == 0:
                v_tree.add((i,j))
            for x in l:
                if x < n[i][j]:
                    tree_count+=1
                elif x >= n[i][j]:
                    tree_count+=1
                    break
            score *= tree_count
            tree_count = 0
            # up 
            l = np.flip(n[:i,j])
            if len(l[l>=n[i][j]]) == 0:
                v_tree.add((i,j))
            for x in l:
                if x < n[i][j]:
                    tree_count+=1
                elif x >= n[i][j]:
                    tree_count+=1
                    break
            score *= tree_count
            tree_count = 0

            # down
            l = n[i+1:,j]
            if len(l[l>=n[i][j]]) == 0:
                v_tree.add((i,j))
            for x in l:
                if x < n[i][j]:
                    tree_count+=1
                elif x >= n[i][j]:
                    tree_count+=1
                    break
            score *= tree_count
            max_score = max(max_score, score)                
    print(f"Part 1 the number of visable trees is {len(v_tree)}")
    print(f"Part 2 the highest possible scenic score is {max_score}")


def _day8_a():
    """
    """
    day = "30373\n25512\n65332\n33549\n35390"
    day = 8
    puzzle = get_input(day, '\n', list)
    n=np.array(puzzle, int)
    v_tree = set()
    max_score = 0
    for i in range(len(n)):
        for j in range(len(n[i])):
            # Boundry checks, tree at the edge is visible for part 1 and 0 for part 2 because of the scoring system.
            if i == 0 or i == len(n[i])-1:
                v_tree.add((i,j))
                continue
            if j == 0 or j == len(n[:,i])-1:
                v_tree.add((i,j))
                continue
            if (i,j) in v_tree:  # We already have seen this tree?
                print(f"({i},{j})")
                continue
            score = 1

            # left
            tree_count = 0
            l = np.flip(n[i][:j])
            w = np.where(l>=n[i][j])[0]
            if w.size == 0:
                v_tree.add((i,j))  # Tree is visible for part 1
                tree_count += j  # Number of trees it can see for part 2
            else:
                tree_count += w[0] + 1
            score *= tree_count
                
            # right
            tree_count = 0
            l = n[i][j+1:]
            w = np.where(l>=n[i][j])[0]
            if w.size == 0:
                v_tree.add((i,j))
                tree_count += len(n[i]) - j - 1
            else:
                tree_count += w[0] + 1
            score *= tree_count
            tree_count = 0

            # up 
            l = np.flip(n[:i,j])
            w = np.where(l>=n[i][j])[0]
            if w.size == 0:
                v_tree.add((i,j))
                tree_count += i
            else:
                tree_count += w[0] + 1
            score *= tree_count
            tree_count = 0

            # down
            l = n[i+1:,j]
            w = np.where(l>=n[i][j])[0]
            if w.size == 0:
                v_tree.add((i,j))
                tree_count += len(n[:,j]) - i - 1
            else:
                tree_count += w[0] + 1
            score *= tree_count
            #print(f"({i},{j}) {n[i][j]} {l} {w} {tree_count}")
            max_score = max(max_score, score)                
    print(f"Part 1 the number of visable trees is {len(v_tree)}")
    print(f"Part 2 the highest possible scenic score is {max_score}")    
    #return n


def plot(position_set):
        """
        Quick function to print out the position set in a more readible form.
        """
        adj = np.array([0,0])
        size = np.array([0,0])
        for i in position_set:
            adj[0] = min(adj[0], i[0])
            adj[1] = min(adj[1], i[1])
            size[0] = max(size[0], i[0])
            size[1] = max(size[1], i[1])
        g = np.full(tuple(size+abs(adj)+[2,2])," ",str)
        for i in position_set:
            adj_p = i + abs(adj)
            g[tuple(adj_p)] = "#"
        g=np.flip(g,0)
        for i in range(len(g[:,0])):
            print("".join(g[i]))


def move_a_knot(h, t):
    """
    Function to update the tail position given the head has moved by one step.
    """
    diff = h - t
    diff_sign = np.sign(diff)  # -1 or 1 depending on the result of h - t
    # Using a manhattan distance calc to decide if we need to move.
    manhattan_diff = np.absolute(diff).sum()
    if manhattan_diff > 2:  # Diagonal move required.
        t += diff_sign
    elif manhattan_diff == 2:
        if diff[1] == 0:  # Vertical move required
            t[0] += diff_sign[0]
        elif diff[0] == 0:  # Horizontal move required
            t[1] += diff_sign[1]
    return t


def _day9(example=False, reload=False):
    """
    Rope physics!
    """
    if example:
        day = "R 4\nU 4\nL 3\nD 1\nR 4\nD 1\nL 5\nR 2"
        day = "R 5\nU 8\nL 8\nD 3\nR 17\nD 10\nL 25\nU 20"
    else:
        day = 9
    puzzle = get_input(day, '\n', lambda x:(x.split()[0], int(x.split()[1])), reload)
    dir_dict = {"U": np.array([ 1,  0]),
                "D": np.array([-1,  0]),
                "R": np.array([ 0,  1]),
                "L": np.array([ 0, -1])}
    p1_position_set = set()
    p2_position_set = set()
    knots = []
    number_of_knots = 10
    p2 = 0
    for _ in itertools.repeat(None, number_of_knots):
        knots.append(np.array([0,0]))
    for direction, motion in puzzle:
        for _ in itertools.repeat(None, motion):
            knots[0] += dir_dict[direction]  # Head is the first in the list.
            for i in range(1, len(knots)):
                knots[i] = move_a_knot(knots[i - 1], knots[i])
            p1_position_set.add(tuple(knots[1]))  # Part 1 tail is just behind the head
            p2_position_set.add(tuple(knots[-1]))  # Part 2 tail is the last in the list.
            p2 += 1
    print(f"For part 1 the tail visits {len(p1_position_set)} locations")
    print(f"For part 2 the tail visits {len(p2_position_set)} locations")
    print(p2)
        # Debug code to see the visited positions.
        # plot(position_set)


def _day10(reload=False):
    """
    CRT monitor simulation.
    """
    puzzle = get_input(10, '\n', None, reload)
    monitor = np.full((6,40)," ")
    x = cycle = 1
    ip = horz = vert = strength = value = 0
    while ip < len(puzzle):
        # Special cycles for part 1.
        if cycle in [20,60,100,140,180,220]:
            strength += (x * cycle)
        # Cycle when the CRT scans to the next line.
        if cycle in [41, 81, 121, 161, 201]:
            vert += 1
            horz = 0
        # CRT update during
        if horz in [x - 1, x, x + 1]:
            monitor[vert][horz] = "#"
        horz += 1
        # End of cycle?
        if value != 0:  # 2nd cycle of a addx command.
            x += value
            value = 0
        else:  # Check to see if the command was addx.
            command = puzzle[ip]
            if command.startswith("addx"):
                value = int(command.split()[1])
            ip += 1  # Move the instruction pointer.
        cycle += 1  # End of the cycle.

    print(f"Part 1 sum of six signal strengths is {strength}")
    print("")
    for i in range(len(monitor[:,0])):
        print("".join(monitor[i]))



def _day11(example=False, part=1, reload=False):
    """
    Stupid Monkeys
    """
    from sympy.ntheory import factorint
    if example:
        day = """Monkey 0:
  Starting items: 79, 98
  Operation: new = old * 19
  Test: divisible by 23
    If true: throw to monkey 2
    If false: throw to monkey 3

Monkey 1:
  Starting items: 54, 65, 75, 74
  Operation: new = old + 6
  Test: divisible by 19
    If true: throw to monkey 2
    If false: throw to monkey 0

Monkey 2:
  Starting items: 79, 60, 97
  Operation: new = old * old
  Test: divisible by 13
    If true: throw to monkey 1
    If false: throw to monkey 3

Monkey 3:
  Starting items: 74
  Operation: new = old + 3
  Test: divisible by 17
    If true: throw to monkey 0
    If false: throw to monkey 1"""
    else:
        day = 11
    puzzle = get_input(day, '\n', None, reload)
    monkey_dict = {}
    current_monkey = None
    for line in puzzle:
        if line.startswith("Monkey"):
            num = int(line.strip(":").split()[1])
            if num in monkey_dict:
                raise Exception()
            else:
                monkey_dict[num]={"inspect":0}
                current_monkey = num
        elif line.lstrip().startswith("Starting"):
            monkey_dict[current_monkey]["items"] = list(map(int, line.split(":")[1].split(',')))
        elif line.lstrip().startswith("Operation"):
            if line.split(" ")[-1] == "old":
                monkey_dict[current_monkey]["operation"] = "** 2"  # hack
            else:
                monkey_dict[current_monkey]["operation"] = line.split("=")[1].strip().replace("old","").strip()
        elif line.lstrip().startswith("Test"):
            monkey_dict[current_monkey]["test"] = int(line.split("by")[1].strip())
        elif line.lstrip().startswith("If true"):
            monkey_dict[current_monkey]["true"] = int(line.split("monkey")[1].strip())
        elif line.lstrip().startswith("If false"):
            monkey_dict[current_monkey]["false"] = int(line.split("monkey")[1].strip())
    # for k,v in monkey_dict.items():
    #     print(f"M{k} - {v}")
    monkey_list = sorted(monkey_dict.keys())
    rounds = 20
    if part == 2:  # just multiply together to get a common multiple of the test divisions and "keep your worry levels manageable"
        modifier = 1
        for monkey in monkey_list:
            modifier *= monkey_dict[monkey]["test"]
        rounds = 10000
    for r in range(rounds):
        for monkey in monkey_list:
            while len(monkey_dict[monkey]["items"]) > 0:
                monkey_dict[monkey]["inspect"] += 1
                item = monkey_dict[monkey]["items"].pop(0)
                new_worry = eval(str(item) + monkey_dict[monkey]["operation"])
                if part == 1:
                    new_worry //= 3
                else:
                    new_worry %= modifier
                if new_worry % monkey_dict[monkey]["test"] == 0:
                    monkey_dict[monkey_dict[monkey]["true"]]["items"].append(new_worry)
                else:
                    monkey_dict[monkey_dict[monkey]["false"]]["items"].append(new_worry)
                # for k,v in monkey_dict.items():
                #     print(k,v)
    # Get the final answer.
    monkey_activity = []
    for m, d in monkey_dict.items():
        monkey_activity.append(d["inspect"])
    monkey_activity.sort(reverse=True)
    print(f"The monkey buisness was {monkey_activity[0] * monkey_activity[1]}")


def bfs(graph, start, end):
    moves_list = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    queue = [(start, 0)]
    visited = set()
    low_score = 41 * 145

    while len(queue) > 0:
        (position, score) = queue.pop(0)
        if position == end:
            if score < low_score:
                low_score = score
            continue
        if position in visited:
            continue
        visited.add(position)
        for move in moves_list:
            next_position = tuple(np.array(position)+move)
            curr_height = graph[position]
            next_height = graph[next_position]
            if curr_height + 1 >= next_height:
                queue.append((next_position, score + 1))
    return low_score


def _day12(example=False, reload=False):
    """
    Walking up the hill, step by step.
    """
    if example:
        day = """Sabqponm\nabcryxxl\naccszExk\nacctuvwj\nabdefghi"""
    else:
        day = 12
    puzzle = get_input(day, "\n", None, reload)
    # Graph, padded with 101 to make the testing easier.
    graph = np.full((len(puzzle) + 2,len(puzzle[0]) + 2), 101, int)
    for i in range(len(puzzle)):
        for j in range(len(puzzle[i])):
            shift_pos = tuple([i + 1, j + 1])
            if puzzle[i][j]=="S":
                graph[shift_pos] = 1
                start = shift_pos
            elif puzzle[i][j] =="E":
                graph[shift_pos] = 26
                end = shift_pos
            else:
                graph[shift_pos] = ord(puzzle[i][j]) - 0x60
    min_score = bfs(graph, start, end)
    print(f'Part 1 the shortest path from "S" to "E" is {min_score} steps')
    print(len(np.argwhere(graph ==1)))
    for start in np.argwhere(graph == 1):
        min_score = min(min_score, bfs(graph, tuple(start), end))
    print(f'Part 2 the shortest path from any "a" to "E" is {min_score} steps')


def packet_compare(left, right):
    """
    Compare two packets.
    -1 if order is correct left, right
     1 if order is not correct
     0 if they are identical.
    """
    result = 0
    for i in range(max(len(left),len(right))):
        # Deal with list that ends early.
        l = left[i] if i < len(left) else None
        r = right[i] if i < len(right) else None
        # Checks to deal with list vs list or list vs int
        if type(l) == int and type(r) == list:
            result = pair_check([l], r)
        elif type(l) == list and type(r) == int:
            result = pair_check(l, [r])
        elif type(l) == list and type(r) == list:
            result = pair_check(l, r)
        elif r is None:  # Right ended first order incorrect
            result = 1
        elif l is None:  # Left ended first, order correct
            result = -1
        elif l < r:  # Left less than right, order correct
            result = -1
        elif l > r:  # Right less than left, order incorrect
            result = 1
        # Made a decision, stop checking.
        if result != 0:
            break
    return result


def _day13(example=False, reload=False):
    """
    Communication packets and sorting!
    """
    import functools
    if example:
        day = """[1,1,3,1,1]\n[1,1,5,1,1]\n\n[[1],[2,3,4]]\n[[1],4]\n\n[9]\n[[8,7,6]]\n\n[[4,4],4,4]\n[[4,4],4,4,4]\n\n[7,7,7,7]\n[7,7,7]\n\n[]\n[3]\n\n[[[]]]\n[[]]\n\n[1,[2,[3,[4,[5,6,7]]]],8,9]\n[1,[2,[3,[4,[5,6,0]]]],8,9]"""
    else:
        day = 13
    puzzle = get_input(day, "\n", None, reload)
    # Parse the puzzle input, eval to the rescue again.
    packets = []
    for line in puzzle:
        if line == "":
            continue
        packets.append(eval(line))
    # Part 1, check pairs of packets.
    score = 0
    index = 1
    for left, right in zip(*[iter(packets)]*2):
        if pair_check(left, right) == -1:
            score += index
        index += 1
    print(f"Part 1 sum of indices is {score}")
    # Part 2 add the divider packets, sort and then find their locations.
    packets.append([[2]])
    packets.append([[6]])
    sorted_packets = sorted(packets, key=functools.cmp_to_key(pair_check))
    i1 = sorted_packets.index([[2]]) + 1
    i2 = sorted_packets.index([[6]]) + 1
    print(f"Part 2 divider packet index multiplication is {i1*i2}")


def go(day=6, time=False):

    try:
        return eval("_day{}".format(day))
    except Exception as e:
        print(e)

