import re
import sys
import math
import time
import copy
import curses
import pickle
import socket
import string
import requests
import functools
import itertools
import statistics
import collections
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


def day1():
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


def day1_eval():
    day = "1000\n2000\n3000\n\n4000\n\n5000\n6000\n\n7000\n8000\n9000\n\n10000\n"
    day = 1
    puzzle = list(get_input(day, '\n\n', None, True))
    puzzle[-1]=puzzle[-1].strip('\n')  # Formatting fix
    elves = [eval(x.replace("\n",'+')) for x in puzzle]
    elves.sort(reverse=True)
    print(f"Single most calories {elves[0]}")
    print(f"Total of the top three elves {sum(elves[:3])}")


def day2_orig():
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


def day2():
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


def day3():
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


def day4():
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


def day5():
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


def day6():
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


def day7():
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


def day8(example=True):
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


def day8_a():
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


def day9(example=False, reload=False):
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


def day10_old(reload=False):
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


def day10_viz(stdscr, puzzle):
    """
    CRT monitor simulation.
    """
    #puzzle = get_input(10, '\n', None, reload)
    stdscr.keypad(True)
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, 8, curses.COLOR_BLACK)
    stdscr.erase()
    stdscr.addstr(1, 0, "Press any key to begin")
    stdscr.getch()
    stdscr.refresh()
    stdscr.erase()
    monitor = np.full((6,41)," ")
    x = cycle = 1
    ip = horz = vert = strength = value = 0
    while ip < len(puzzle):
        # Draw the 3 characters around x
        for i in [-1,0,1]:
            if monitor[vert][x+i] != "#":  # Don't overwrite the green #
                try:
                    stdscr.addch(vert, x+i, '\u2592', curses.color_pair(2))
                except:
                    foo=1
        # Draw a dot for the current raster point
        stdscr.addch(vert, horz, chr(0xB7), curses.color_pair(1))
        # Show the screen.
        stdscr.refresh()
        time.sleep(0.05)
        # Clear the 3 characters around x and clear the dot.
        stdscr.addch(vert, horz, " ", curses.color_pair(1))
        for i in [-1,0,1]:
            if monitor[vert][x+i] != "#":
                try:
                    stdscr.addch(vert, x+i, " ", curses.color_pair(1))
                except:
                    foo=1  # offscreen
        # Special cycles for part 1.
        if cycle in [20,60,100,140,180,220]:
            strength += (x * cycle)
        # Cycle when the CRT scans to the next line.
        if cycle in [41, 81, 121, 161, 201]:
            vert += 1
            horz = 0
        # CRT update during
        if horz in [x - 1, x, x + 1]:
            # Draw a green # at the correct spot.
            stdscr.addstr(vert, horz, "#", curses.color_pair(1))
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

    stdscr.addstr(vert + 2, 0, "Press any key to exit")
    stdscr.refresh()
    stdscr.getch()


def day10(reload=False):
    puzzle = get_input(10, '\n', None, reload)
    curses.wrapper(day10_viz, puzzle)

def day11(example=False, part=1, reload=False):
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


def day12(example=False, reload=False):
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
    for i in range(max(len(left), len(right))):
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


def day13(example=False, reload=False):
    """
    Communication packets and sorting!
    """
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
    sorted_packets = sorted(packets, key=functools.cmp_to_key(packet_compare))
    i1 = sorted_packets.index([[2]]) + 1
    i2 = sorted_packets.index([[6]]) + 1
    print(f"Part 2 divider packet index multiplication is {i1*i2}")


def day14(part=1, example=False, reload=False):
    """
    Like grains of sand through the hourglass so are the days of our lives....
    """
    if example:
        day = """498,4 -> 498,6 -> 496,6\n503,4 -> 502,4 -> 502,9 -> 494,9"""
    else:
        day = 14
    puzzle = get_input(day, "\n", None, reload)
    line_list = []
    # Not super happy with this parsing....
    for line in puzzle:
        coords = line.split(" -> ")
        prev = list(map(int, coords[0].split(",")))
        prev.reverse()
        for i in range(1, len(coords)):
            c = list(map(int, coords[i].split(",")))
            c.reverse()
            line_list.append([prev, c])
            prev  = c
    max_y = max_x = 0
    for i in line_list:
        for j in i:
            max_y = max(j[0], max_y)
            max_x = max(j[1], max_x)
    # print(max_x, max_y)  # This was more useful in the example puzzle.
    cave = np.full((max_y+5, 1001), " ", str)  # Hack, because the worse case is the sand forming a triangle with 45 degree angles at the base.
    if part == 2:
        cave[max_y + 2, :]="#"  # Insert the floor for part 2.
    # Fill in the rocks.
    for s,e in line_list:
        y = sorted([s[0], e[0]])
        x = sorted([s[1], e[1]])
        cave[y[0]:y[1] + 1, x[0]:x[1] + 1] = "#"

    sand = np.array([0, 500])
    sand_counter = 0
    while True:
        next_sand = sand + [1, 0]
        if next_sand[0] > max_y + 3:
            print("The Abyss!")
            break
        if cave[tuple(next_sand)] == "#":  # Rock or sand below
            if cave[next_sand[0],next_sand[1] - 1] == "#" and cave[next_sand[0], next_sand[1] + 1] == "#":  # Stop
                cave[tuple(sand)]="#"
                sand_counter += 1
                if sand[0] == 0 and sand[1] == 500:  # Part 2 stop check.
                    print("Stopped the sand at the source")
                    break
                next_sand = np.array([0, 500])  # Reset for the next piece of sand.
            elif cave[next_sand[0], next_sand[1] - 1] != "#":  # Clear to the left
                next_sand += [0, -1]
            elif cave[next_sand[0], next_sand[1] + 1] != "#":  # Clear to the right
                next_sand += [0, 1]
        sand = next_sand  # Sand moved down/diagonal.
        # Debug print for example
        # for r in range(10):
        #     print("".join(cave[r][494:506]))
    print(f"{sand_counter} grains of sand have fallen")


def day15(example=False, reload=False):
    """
    Brute force for part 1 and don't look back....
    """
    day = 15 if example is False else ("Sensor at x=2, y=18: closest beacon is at x=-2, y=15\n"
                                       "Sensor at x=9, y=16: closest beacon is at x=10, y=16\n"
                                       "Sensor at x=13, y=2: closest beacon is at x=15, y=3\n"
                                       "Sensor at x=12, y=14: closest beacon is at x=10, y=16\n"
                                       "Sensor at x=10, y=20: closest beacon is at x=10, y=16\n"
                                       "Sensor at x=14, y=17: closest beacon is at x=10, y=16\n"
                                       "Sensor at x=8, y=7: closest beacon is at x=2, y=10\n"
                                       "Sensor at x=2, y=0: closest beacon is at x=2, y=10\n"
                                       "Sensor at x=0, y=11: closest beacon is at x=2, y=10\n"
                                       "Sensor at x=20, y=14: closest beacon is at x=25, y=17\n"
                                       "Sensor at x=17, y=20: closest beacon is at x=21, y=22\n"
                                       "Sensor at x=16, y=7: closest beacon is at x=15, y=3\n"
                                       "Sensor at x=14, y=3: closest beacon is at x=15, y=3\n"
                                       "Sensor at x=20, y=1: closest beacon is at x=15, y=3")
    puzzle = get_input(day, "\n", None, reload)
    s_d = {}
    max_x = 0
    min_x = 10000
    for line in puzzle:
        sx = int(line.split(',')[0].split("=")[1])
        sy = int(line.split(':')[0].split("y")[1].strip("="))
        bx = int(line.split("is at")[1].split(",")[0].split("=")[1])
        by = int(line.split("=")[-1])
        s = tuple([sx,sy])
        b=tuple([bx,by])
        m=abs(s[0]-b[0])+abs(s[1]-b[1])
        #print(line, s, b)
        if s in s_d.keys():
            raise Exception(f"Duplicate {s}")
        s_d[s] = {"b":b, "m":m}
        max_x = max(max_x, s[0]+m)
        min_x = min(min_x, s[0]-m)
    print(min_x, max_x)
    y = 10 if example else 2000000
    to_check = []
    for s,b in s_d.items():
        if abs(s[1]-y) <= b['m']:
            to_check.append(s)
    positions = set()
    for x in range(min_x, max_x+1):
        #for s,b in s_d.items():
        for s in to_check:
            m = abs(s[0]-x)+abs(s[1]-y)
            if s_d[s]['b'] == tuple([x,y]):
                continue
            if m <= s_d[s]["m"]:
                #print(x)
                positions.add(tuple([x,y]))
    print(len(positions))


def day15_p2(example=False, reload=False):
    """
    Solution for part 2, still probably some optimization that can be done but manhattan geomertry is annoying.
    """
    day = 15 if example is False else ("Sensor at x=2, y=18: closest beacon is at x=-2, y=15\n"
                                       "Sensor at x=9, y=16: closest beacon is at x=10, y=16\n"
                                       "Sensor at x=13, y=2: closest beacon is at x=15, y=3\n"
                                       "Sensor at x=12, y=14: closest beacon is at x=10, y=16\n"
                                       "Sensor at x=10, y=20: closest beacon is at x=10, y=16\n"
                                       "Sensor at x=14, y=17: closest beacon is at x=10, y=16\n"
                                       "Sensor at x=8, y=7: closest beacon is at x=2, y=10\n"
                                       "Sensor at x=2, y=0: closest beacon is at x=2, y=10\n"
                                       "Sensor at x=0, y=11: closest beacon is at x=2, y=10\n"
                                       "Sensor at x=20, y=14: closest beacon is at x=25, y=17\n"
                                       "Sensor at x=17, y=20: closest beacon is at x=21, y=22\n"
                                       "Sensor at x=16, y=7: closest beacon is at x=15, y=3\n"
                                       "Sensor at x=14, y=3: closest beacon is at x=15, y=3\n"
                                       "Sensor at x=20, y=1: closest beacon is at x=15, y=3")
    puzzle = get_input(day, "\n", None, reload)
    sensor_dict = {}
    no_b = set()
    if example:
        max_i = 20
    else:
        max_i = 4000000
    for line in puzzle:
        sx = int(line.split(',')[0].split("=")[1])
        sy = int(line.split(':')[0].split("y")[1].strip("="))
        bx = int(line.split("is at")[1].split(",")[0].split("=")[1])
        by = int(line.split("=")[-1])
        s = tuple([sx, sy])
        b = tuple([bx, by])
        m = manhattan(s, b)
        if s in sensor_dict.keys():
            raise Exception(f"Duplicate {s}")
        sensor_dict[s] = m

    # Search all pairs of sensors for ones where there is a gap between their coverage just big enough for one becon.
    points_to_check = set()
    lines = []
    for a, b in itertools.combinations(sensor_dict.keys(), 2):
        a_to_b_manhattan = manhattan(a,b)
        d = a_to_b_manhattan - sensor_dict[a] - sensor_dict[b] 
        this_edge = set()
        if d == 2:
            ax,ay = a
            bx,by=b
            #print(f"{a} [{sensor_dict[a]}] -> {b} [{sensor_dict[b]}] d={d} m={m}")
            for dx in range(sensor_dict[a] + 2):  # +1 to the sensor range (+1 for python range non-inclusive)
                if by < ay and bx > ax:  # up right
                    x = ax + dx
                    y = ay - (sensor_dict[a] + 1 - dx)
                elif by < ay and bx < ax:  # up left
                    x = ax - dx
                    y = ay - (sensor_dict[a] + 1 - dx)
                elif by > ay and bx < ax:  # down left
                    x = ax - dx
                    y = ay + (sensor_dict[a] + 1 - dx)
                else:  # down right
                    x = ax + dx
                    y = ay + (sensor_dict[a] + 1 - dx)
                this_edge.add(tuple([x,y]))
            lines.append(this_edge)

    # There should be two edges which overlap at the senor point.
    # Use intersection to reduce the number of points to check.
    points_to_check = set()
    for one, two in itertools.combinations(lines, 2):
        points_to_check = points_to_check.union(set.intersection(one, two))
    print(f"Checking {len(points_to_check)} points")
    # Verify the point is correct by testing against all the sensor coverages
    for point in points_to_check:
        found = True
        for sensor, s_manhattan in sensor_dict.items():
            if manhattan(point, sensor) <= s_manhattan:
                found = False
                break
        if found is True:
            print(point)
            break
    print(f"Part2 frequency is {(point[0]*4000000)+point[1]}")


def valve_dfs(valves, valve, remaining_time, on, pressure, pressures):
    pressures[on] = max(pressures.get(on, 0), pressure)
    for k,v in valves.items():
        new_time = remaining_time - valves[valve]['distance'][k] - 1
        if v['one_hot'] & on or new_time < 0 or v['rate'] == 0:
            continue
        valve_dfs(valves, k, new_time, on | v['one_hot'], pressure + new_time * v['rate'], pressures)
    return pressure



def day16(example=False, reload=False):
    """
    """
    day = 16
    if example:
        day = ("Valve AA has flow rate=0; tunnels lead to valves DD, II, BB\n"
               "Valve BB has flow rate=13; tunnels lead to valves CC, AA\n"
               "Valve CC has flow rate=2; tunnels lead to valves DD, BB\n"
               "Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE\n"
               "Valve EE has flow rate=3; tunnels lead to valves FF, DD\n"
               "Valve FF has flow rate=0; tunnels lead to valves EE, GG\n"
               "Valve GG has flow rate=0; tunnels lead to valves FF, HH\n"
               "Valve HH has flow rate=22; tunnel leads to valve GG\n"
               "Valve II has flow rate=0; tunnels lead to valves AA, JJ\n"
               "Valve JJ has flow rate=21; tunnel leads to valve II")
    puzzle = get_input(day, '\n', None, reload)
    valves = {}
    distance = {}
    valve_one_hot = 1  # Need a hashable way to track which valves were on.
    for line in puzzle:
        name = line.split(" ")[1]
        rate = int(line.split(";")[0].split("=")[1])
        paths = line.split("valve")[1]
        paths = paths.lstrip("s ")
        paths = paths.split(", ")
        if name in valves:
            raise Exception(f"{name} was a duplicate in {valves}")
        valves[name] = {"rate":rate, "paths":paths, "distance":{}, "one_hot":valve_one_hot}
        valve_one_hot = valve_one_hot << 1
    # Calculate the distance between the vales.
    for name, valve in valves.items():
        valve['distance'] = dict.fromkeys(valves.keys(), 1e6)
        for p in valve['paths']:
            valve['distance'][p] = 1
    for i in valves.keys():
        for j in valves.keys():
            for k in valves.keys():
                valves[j]['distance'][k] = min(valves[j]['distance'][k], valves[j]['distance'][i]+valves[i]['distance'][k])
    #print(valves)
    pressures = {}
    valve_dfs(valves, "AA", 30, 0, 0, pressures)
    print(f"Part 1 maximum pressure release {max(pressures.values())}")
    pressures = {}
    visited = valve_dfs(valves, "AA", 26, 0, 0, pressures)
    max_pressure = 0
    for me, my_pressure in pressures.items():
        for ele, ele_pressure in pressures.items():
            if me & ele == 0:  # No overlapping valves
                max_pressure = max(max_pressure, my_pressure + ele_pressure)
    print(f"Part 2 maximum pressure working together is {max_pressure}")


# def push(board, shape, move):
#     new_shape = []
#     for rock in shape:
#         this_rock = rock + move
#         if this_rock[0] in [0, 8]:
#             return shape  # moved past the edge, abort the move
#         else:
#             new_shape.append(rock + move)
#     if move != [0,-1]:  # Air push
#         if set(new_shape).isdisjoint(board) is False: # Cant push sideways)
#             return shape
#     return new_shape
        
        
# def draw_b(board, this_shape):
#     viz = np.full((20,8)," ")
#     for i in board:
#         viz[tuple(reversed(i))]="#"
#     for i in this_shape:
#         viz[tuple(reversed(i))]="#"
#     for y in range(19,-1,-1):
#         print("".join(viz[y]))
#         


# def _draw_top(board, max_y):
#     viz = np.full((21,8), " ")
#     this_y = max_y
#     for i in range(20):
#         for rock in board:
#             if rock[1] == this_y:
#                 viz[(i, rock[0])] = "#"
#         this_y -= 1
#     for y in range(20):
#         print("".join(viz[y]))


# def day17(example=False, reload=False):
#     day = 17
#     if example:
#         day = ">>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>"
#     pushes = get_input(day, "\n", list, reload)[0]
#     shapes = [
#         [(3,0), (4,0), (5,0), (6,0)],  # horz. line
#         [(4,2), (3,1), (4,1), (5,1), (4,0)],  # +
#         [(5,2), (5,1), (3,0), (4,0), (5,0)],  # L
#         [(3,3), (3,2), (3,1), (3,0)],  # vert. line
#         [(3,1), (4,1), (3,0), (4,0)],
#         ]
#     for i in range(len(shapes)):
#         for j in range(len(shapes[i])):
#             shapes[i][j] = Special_Tuple(shapes[i][j])
#     decode = {"<":[-1,0], ">":[1,0]}
#     for i in range(len(pushes)):
#         pushes[i] = decode[pushes[i]]
# 
#     push_index = 0
#     board = set([(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0)])
#     down = [0,-1]
#     max_y = 0
#     shape_no = 1
#     shape_index = 0
#     keep_size = 40
#     cycles = {}
#     for shape in itertools.cycle(shapes):
#         #if shape_no == 2023:
#         if shape_no == 1000000000000+1:
#             break
#         
#         if max_y > keep_size:
#             for x in range(1,8):
#                 for y in range(4):
#                     board.discard(tuple([x,max_y-keep_size-y]))
# 
#         this_shape = []
#         for rock in shape:
#             this_shape.append(rock + [0, max_y + 4])
#         #print(this_shape)
#         while True:
#             # Air moves it
#                 #input()            
#             new_shape = push(board, this_shape, pushes[push_index%len(pushes)])
#             push_index += 1
#             if set(new_shape).isdisjoint(board) is False:
#                 board = set.union(set(this_shape), board)
#                 max_y = max(max_y, max(list(zip(*board))[1]))
#                 break
#             #draw_b(board, this_shape)
#             #print(f"push {shape_no}")
#             #input()
#             # Falls
#             this_shape = new_shape
#             new_shape = push(board, this_shape, [0,-1])
#             if set(new_shape).isdisjoint(board) is False:
#                 board = set.union(set(this_shape), board)
#                 max_y = max(max_y, max(list(zip(*board))[1]))
#                 break
#             this_shape = new_shape
#         #_draw_top(set.union(board, set(new_shape)), max_y)
#         print(f"shape:{shape_no}, max:{max_y} cycle={len(shapes)*len(pushes)} board_size:{len(board)} ")
#         if tuple(sorted(board)) in cycles.keys():
#             print(f"repeat of {cycles[board]}")
#             raise Exception()
#         else:
#             cycles[tuple(sorted(board))] = shape_no
#         #draw_b(board, this_shape)
#         #print(f"shape:{shape_no}  {shape_no % len(shapes)} {push_index % len(pushes)} board size:{len(board)}")
# 
#         #input()
#         shape_no +=1
#         shape_index += 1
#     print(shape_no, max_y)


class Special_Tuple(tuple):
    def __add__(self, other):
        return Special_Tuple(x + y for x, y in zip(self, other))


def draw_b2(board, this_shape):
    """
    Debug print to draw the tetris board.
    """
    viz = np.full((40,8)," ")
    for i in board:
        viz[tuple(reversed(i))]="#"
    for i in this_shape:
        viz[tuple(reversed(i))]="#"
    for y in range(40):
        print("".join(viz[y]))


def move_d17(board, shape=None, move=-1):
    """
    Move the tetris shape or the board
    """
    new = set()
    if shape is None:  # Move the board
        for rock in board:
            new.add(rock + [0, move])
    else:  # Push the shape
        for rock in shape:
            new_rock = rock + move
            if new_rock[0] in [0, 8]:  # Moved past the edge, abort the move
                new = shape
                break
            else:
                new.add(new_rock)
        if new.isdisjoint(board) is False:  # Can't push sideways something is in the way
            new = shape
    return new


def day17(part=1, example=False, reload=False):
    day = 17
    if example:
        day = ">>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>"
    pushes = get_input(day, "\n", list, reload)[0]
    shapes = [
        [1, set([Special_Tuple((3,0)), Special_Tuple((4,0)), Special_Tuple((5,0)), Special_Tuple((6,0))])],  # horz. line
        [3, set([Special_Tuple((4,2)), Special_Tuple((3,1)), Special_Tuple((4,1)), Special_Tuple((5,1)), Special_Tuple((4,0))])],  # +
        [3, set([Special_Tuple((5,0)), Special_Tuple((5,1)), Special_Tuple((3,2)), Special_Tuple((4,2)), Special_Tuple((5,2))])],  # L
        [4, set([Special_Tuple((3,3)), Special_Tuple((3,2)), Special_Tuple((3,1)), Special_Tuple((3,0))])],  # vert. line
        [2, set([Special_Tuple((3,1)), Special_Tuple((4,1)), Special_Tuple((3,0)), Special_Tuple((4,0))])],
        ]

    decode = {"<":[-1,0], ">":[1,0]}
    for i in range(len(pushes)):
        pushes[i] = decode[pushes[i]]

    board = set([Special_Tuple((1,0)),
                 Special_Tuple((2,0)),
                 Special_Tuple((3,0)),
                 Special_Tuple((4,0)),
                 Special_Tuple((5,0)),
                 Special_Tuple((6,0)),
                 Special_Tuple((7,0))])

    shape_no = 0
    push_index = 0
    max_y = 0
    cycles = {}
    tower = 2022 if part == 1 else 1000000000000
    finish = tower
    found = False
    cycle_hight = None
    no_of_cycles = None

    for this_shape in itertools.cycle(shapes):

        # Correction for when the piece drops below the top of the tower.
        min_y = min(0, min(list(zip(*board))[1]))
        if min_y != 0:
            board = move_d17(board, None, abs(min_y))

        # Erase the bottom of the board to speed the set checks.
        for y in range(40, max(list(zip(*board))[1])+1):
            max_y += 1
            for x in range(8):
                board.discard((x,y))

        if shape_no == finish:
            break
        size, shape = this_shape

        # Logic to look for the cyclic nature of the puzzle.
        if shape_no % len(shapes) == 0 and part == 2 and not found:
            t_board = tuple(board)
            if t_board in cycles.keys():
                #draw_b2(board, set())
                print(f"Duplicate [{shape_no}, {max_y+max(list(zip(*board))[1])}] with {cycles[t_board]}")
                cycle_shape_count = shape_no - cycles[t_board][0]
                cycle_height = max_y+max(list(zip(*board))[1]) - cycles[t_board][1]
                print(f" Cycle repeats every {cycle_shape_count} shapes and grows by {cycle_height}")
                finish = shape_no + (tower-shape_no) % cycle_shape_count
                print(f" Continue to shape {finish} to get the remainder height")
                no_of_cycles = (tower-finish)//cycle_shape_count
                found = True
            else:
                cycles[t_board] = [shape_no, max_y+max(list(zip(*board))[1])]

        # Move board down to fit the new shape
        board = move_d17(board, None, size + 3)

        while True:
            # Air moves it
            shape = move_d17(board, shape, pushes[push_index%len(pushes)])
            push_index += 1

            # Shape falls (really the board moves up so that the top of the shape stays at y==0)
            new_board = mov_board(board)

            # Check for overlap which would stop the piece
            if shape.isdisjoint(new_board) is False:
                board = set.union(shape, board)
                break
            board = new_board
        # Next shape
        shape_no +=1

    # Final prints.
    cur_max_y = max_y+max(list(zip(*board))[1])
    if part == 1:
        print(f"Part 1 answer is {cur_max_y}")
    else:
        print(f"Current max_y:{cur_max_y} at {shape_no}")
        cycle_y = no_of_cycles * cycle_height
        print(f" Number of cycles {no_of_cycles} shape_count = {cycle_shape_count}")
        print(f" Additional shapes {no_of_cycles*cycle_shape_count} + {shape_no} = {(no_of_cycles*cycle_shape_count)+shape_no}")
        print(f" Add {cycle_y} for {no_of_cycles} cycles each adding {cycle_height} height")
        ans = cycle_y + cur_max_y
        print(f"Part 2 answer is {ans}")
        if example:
            print(f" Check example agains known value: {ans-1514285714288}")


def day18(example=False, reload=False):
    day = 18
    if example:
        day = ("2,2,2\n"
               "1,2,2\n"
               "3,2,2\n"
               "2,1,2\n"
               "2,3,2\n"
               "2,2,1\n"
               "2,2,3\n"
               "2,2,4\n"
               "2,2,6\n"
               "1,2,5\n"
               "3,2,5\n"
               "2,1,5\n"
               "2,3,5")
        #day = "1,1,1\n2,1,1"
    puzzle = get_input(day, '\n', lambda x: list(map(int, x.split(","))), reload)
    cube_faces = []
    for x,y,z in puzzle:
        a = (x,  y,  z,   x+1, y+1, z)
        b = (x,  y+1,z,   x+1, y+1, z+1)
        c = (x+1,y,  z,   x+1, y+1, z+1)
        d = (x,  y,  z,   x+1, y,   z+1)
        e = (x,  y,  z,   x,   y+1, z+1)
        f = (x,  y,  z+1, x+1, y+1, z+1)
        cube_face = tuple([a,b,c,d,e,f])
        cube_faces.append(cube_face)
    pos_faces = 0
    neg_faces = 0
    face_set = set()
    for cube in cube_faces:
        for face in cube:
            if face in face_set:
                neg_faces += 2
            pos_faces += 1
            face_set.add(face)
    print(pos_faces-neg_faces)


def _bfs_outside(x,y,z, max_pos, cube_set):
    queue = []
    visited = set()
    if (x,y,z) not in visited and (x,y,z) not in cube_set:
        queue.append((x,y,z))
        visited.add((x,y,z))

    while len(queue) > 0:
        x,y,z = queue.pop(0)
        if min(x,y,z) < 0 or x >= max_pos[0] or y >= max_pos[1] or z >= max_pos[2]:
            return False
        for i in [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]:
            t = (x+i[0],y+i[1],z+i[2])
            if t not in visited and t not in cube_set:
                queue.append(t)
                visited.add(t)
    return True
        
def day18_p2(example=False, reload=False):
    #global cube_set
    day = 18
    if example:
        day = ("2,2,2\n"
               "1,2,2\n"
               "3,2,2\n"
               "2,1,2\n"
               "2,3,2\n"
               "2,2,1\n"
               "2,2,3\n"
               "2,2,4\n"
               "2,2,6\n"
               "1,2,5\n"
               "3,2,5\n"
               "2,1,5\n"
               "2,3,5")
    puzzle = get_input(day, '\n', lambda x: tuple(map(int, x.split(","))), reload)
    x_pos, y_pos, z_pos = list(zip(*puzzle))
    max_x = max(x_pos)+1
    max_y = max(y_pos)+1
    max_z = max(z_pos)+1
    #print(f"max_x:{max_x} max_y:{max_y} max_z:{max_z}")
    cube_set = set(puzzle)
   
    enclosed_set = set()
    max_pos = (max_x+1, max_y+1, max_z+1)
    for x in range(1,max_x):
        for y in range(1, max_y):
            for z in range(1, max_z):
                if (x,y,z) in puzzle:
                    continue
                #print(x,y,z)
                if _bfs_outside(x,y,z, max_pos, cube_set):
                    enclosed_set.add(tuple([x,y,z]))
    size = 0
    for cube in puzzle:
        for i in [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]:
            t = (cube[0]+i[0],cube[1]+i[1],cube[2]+i[2])
            if t not in puzzle and t not in enclosed_set:
                size+=1
    print(size)


def day19(example=False, reload=False):
    """
    """
    day = 19
    if example:
        day = ("Blueprint 1: Each ore robot costs 4 ore. Each clay robot costs 2 ore. Each obsidian robot costs 3 ore and 14 clay. Each geode robot costs 2 ore and 7 obsidian.\n"
               "Blueprint 2: Each ore robot costs 2 ore. Each clay robot costs 3 ore. Each obsidian robot costs 3 ore and 8 clay. Each geode robot costs 3 ore and 12 obsidian.")
    puzzle = get_input(day, "\n", None, reload)

    blueprint_dict = {}
    for line in puzzle:
        num = int(line.split("Blueprint")[1].split(":")[0])
        ore = int(line.split("ore.")[0].split("costs")[1])
        clay = int(line.split("ore.")[1].split("costs")[1])
        obsidion_o = int(line.split("ore")[3].split("costs")[1])
        obsidion_c = int(line.split("clay")[1].split("and")[1])
        geode_o = int(line.split("ore")[4].split("costs")[1])
        geode_b = int(line.split("obsidian")[1].split("and")[2])
        if num in blueprint_dict.keys():
            raise Exception(f"Duplicate {num} in {blueprint_dict.keys()}")
        blueprint_dict[num] = {"ore":  {"ore": ore},
                               "clay": {"ore": clay},
                               "obsidian": {"ore": obsidion_o, "clay": obsidion_c},
                               "geode": {"ore": geode_o, "obsidian": geode_b}}
    print(blueprint_dict)


def day19(example=False, reload=False):
    day = 19
    if example:
        day = ""
    puzzle = get_input(day, "\n", None, reload)


def print_q(message, value_dict):
    """
    Helper function to debug day 20 printing the original message to check against the example.
    """
    temp_list = []
    for i in message:
        temp_list.append(value_dict[i])
    #temp_list.insert(0, temp_list.pop(-1))
    print(temp_list)


def day20(part=1, example=False, reload=False):
    """
    Encrypted messages, tried to use a simple list at first but had not considered there
    might be duplicate values in the message.
    """
    day=20
    if example:
        day = "1\n2\n-3\n3\n-2\n0\n4"
    original_file = get_input(day, "\n", int, reload)
    value_dict = {}
    message = collections.deque(maxlen=len(original_file))
    decryption = 811589153
    where_is_zero = None
    # Parse the puzzle and load a dictionary with all the values giving each a unique key (index)
    for index, number in enumerate(original_file):
        if index in value_dict.keys():
            raise Exception("duplicate {index}")
        if number == 0:
            where_is_zero = index
        value_dict[index] = number if part == 1 else number * decryption
        message.append(index)
    print("Zero is at ", where_is_zero)  # Keep track of where 0 is because it is important later.

    # Start the message decryption
    print("Start mixing")
    mixing_loop = 1 if part == 1 else 10
    for i in range(mixing_loop):
        for index, value in value_dict.items():
            # Search for the number by finding it's associated key (index) and
            # moving it to the end of the queue
            message.rotate((message.index(index) + 1) * -1) # + 1 to move it to the bottom instead of the top.
            message.pop()
            message.rotate(value * -1)  # Rotate by the desired amount then re-insert
            message.append(index)

    # Look up the grove coordinates
    answer = 0
    for check in [1000, 2000, 3000]:
        print(f"Checking {check}")
        message.rotate(message.index(where_is_zero) * -1)  # Start at 0
        message.rotate(-1 * check)  # Rotate 1000, 2000, or 3000 times.
        answer += value_dict[message[0]]
    print(answer)


def _monkey_math(monkey, monkeys, humn_req=False):
    if monkey == "humn":
        humn_req = True
    if type(monkeys[monkey]["op"]) in [int, float]:
        return monkeys[monkey]["op"], humn_req
    else:
        left, humn_req_l = _monkey_math(monkeys[monkey]["left"], monkeys, humn_req)
        right, humn_req_r = _monkey_math(monkeys[monkey]["right"], monkeys, humn_req)
        return monkeys[monkey]["op"](left,right), humn_req_l | humn_req_r


def _dfs_path(monkeys, start, end):
    stack = [(start, [start])]
    visited = set()
    while stack:
        (monkey, path) = stack.pop()
        if monkey not in visited:
            if monkey == end:
                return path
            visited.add(monkey)
            if type(monkeys[monkey]["op"]) is not int:
                stack.append((monkeys[monkey]["left"], path+[monkeys[monkey]["left"]]))
                stack.append((monkeys[monkey]["right"], path+[monkeys[monkey]["right"]]))


def day21(example=False, reload=False, debug_print=False):
    """
    """
    import operator
    day=21

    # Helper lookup tables.
    reverse = {operator.add: operator.sub,
               operator.sub: operator.add,
               operator.mul: operator.ifloordiv,
               operator.ifloordiv: operator.mul}
    decode = {"+": operator.add,
              "-": operator.sub,
              "*": operator.mul,
              "/": operator.ifloordiv}
    debug_lookup = {v: k for k, v in decode.items()}  # Only for debug prints

    if example:
        day = ("root: pppw + sjmn\n"
               "dbpl: 5\n"
               "cczh: sllz + lgvd\n"
               "zczc: 2\n"
               "ptdq: humn - dvpt\n"
               "dvpt: 3\n"
               "lfqf: 4\n"
               "humn: 5\n"
               "ljgn: 2\n"
               "sjmn: drzm * dbpl\n"
               "sllz: 4\n"
               "pppw: cczh / lfqf\n"
               "lgvd: ljgn * ptdq\n"
               "drzm: hmdt - zczc\n"
               "hmdt: 32\n")
    puzzle = get_input(day, '\n', None, reload)
    monkeys = {}
    for line in puzzle:
        monkey_name = line.split(":")[0]
        if monkey_name in monkeys.keys():
            raise Exception("Duplicate monkey {monkey_name} in {monkey_dict}")
        line_list = line.split()
        monkeys[monkey_name] = {}
        if len(line_list) == 2:
            monkeys[monkey_name] = {"op": int(line_list[-1])}
        else:
            monkeys[monkey_name] = {"op": decode[line_list[2]]}
            monkeys[monkey_name]["left"]  = line_list[1]
            monkeys[monkey_name]["right"] = line_list[3]

    print(f'Part 1\n The "root" monkey will yell {_monkey_math("root", monkeys)[0]}')

    # Collapse the graph of independent monkeys (the ones that don't depend on humn)
    for monkey in monkeys.keys():
        value, humn_req = _monkey_math(monkey, monkeys)
        if not humn_req:
            monkeys[monkey] = {"op": value}

    print("Part 2")
    # Decide what the value to match should be.
    to_match=None
    for side in ["right", "left"]:
        value, humn_req = _monkey_math(monkeys["root"][side], monkeys)
        if not humn_req:
            to_match = value
            other = {"left":"right", "right":"left"}[side]
            print(f' {side.capitalize()} monkey "{monkeys["root"][side]}" is independent, value to match is {to_match}')
            
    path = _dfs_path(monkeys, monkeys["root"][other], "humn")
    # print(f'Path from {monkeys["root"][other]} to humn is {path}')
    path.pop()  # Remove humn

    # Reverse the math operations while moving to humn
    for monkey in path:
        left = monkeys[monkey]["left"]
        right = monkeys[monkey]["right"]
        value_l,  humn_req_l =  _monkey_math(monkeys[monkey]["left"], monkeys)
        value_r, humn_req_r =  _monkey_math(monkeys[monkey]["right"], monkeys)
        op = monkeys[monkey]["op"]
        rev_op = reverse[op]
        if debug_print:
            if humn_req_l:
                print(f" {monkey} = {to_match} = {left} {debug_lookup[op]} {value_r}")
            else:
                print(f" {monkey} = {to_match} = {value_l} {debug_lookup[op]} {right}")
        if humn_req_l and humn_req_r:  # Really hope this does not happen
            raise Exception("Both sides depend on humn!")
        if humn_req_l:  # Solve the equation for the left side monkey
            solve = left
            name_order = [monkey, right]
            value_order = [to_match, value_r]
        else:  # Solve the equation for the right side monkey.
            solve = right
            name_order = [monkey, left]
            value_order = [to_match, value_l]
        if humn_req_r and op in [operator.sub, operator.ifloordiv]:  # Special case for solving the right hand side.
            rev_op = reverse[rev_op]
            value_order.reverse()
        #print(f" {solve} = {name_order[0]} {debug_lookup[rev_op]} {name_order[1]}")
        to_match = rev_op(value_order[0], value_order[1])
        if debug_print:
            print(f" {solve} = {value_order[0]} {debug_lookup[rev_op]} {value_order[1]} = {to_match}")
    # Double check the answer.
    monkeys["humn"]["op"] = to_match
    left, _ = _monkey_math(monkeys["root"]["left"], monkeys)
    right, _ = _monkey_math(monkeys["root"]["right"], monkeys)
    if left == right:
        print(f' humn = {to_match} makes {monkeys["root"]["left"]} = {monkeys["root"]["right"]} = {left}')


def go(day=6, time=False):

    try:
        return eval("day{}".format(day))
    except Exception as e:
        print(e)

