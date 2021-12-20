import requests
import numpy
import pickle
from os import path
from functools import reduce
import itertools

# The Value from the session cookie used to make the webaccess.
# You could hardcode this with your value or set it at the interactive prompt.
# This is because I am lazy and didn't want to figure out how to scrape the cookie or work with the OAuth.
# I'd never work on these at the office but...
_code_path = r'c:\AoC'
#_code_path = r'/mnt/f/aoc/'
_code_path =  r'c:\aoc'
_unix = False
_work = False
_offline = False
_year = 2020

def _pull_puzzle_input(day, seperator, cast):
    """
    Pull the puzzle data from the AOC website.

    :param day: integer day value
    :param seperator: string the data seperator for the data
    :param cast: function to call on each item in the list

    :return: tuple of the data.
    """
    global _work, _offline, _code_path

    if _offline:

        with open(_code_path + r"\{}\day{}.txt".format(_year, day)) as file_handler:
           data_list = file_handler.read().split(seperator)
    else:
        if not path.exists(_code_path + "/session.txt"):
            raise Exception("Using the web browser get the session cookie value\nand put it as a string in {}".format(_code_path + "\session.txt"))
        with open(_code_path + "/session.txt", 'r') as session_file:
            session = session_file.read()
        if _work:
            proxy_dict = {'http': 'proxy-dmz.intel.com:911',
                          'https': 'proxy-dmz.intel.com:912'}
        else:
            proxy_dict = {}
        header = {'Cookie' : 'session={:s}'.format(session.rstrip("\n"))}
        with requests.Session() as session:
            resp = session.get('https://adventofcode.com/{}/day/{}/input'.format(_year, day), headers = header, proxies = proxy_dict)
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
    Wrapper for the global dict.
    """
    global _code_path
    if path.exists(_code_path + r'\{}\input.p'.format(_year)):
        puzzle_dict = pickle.load(open(_code_path + r'\{}\input.p'.format(_year), 'rb'))
    else:  # No pickle file, will need to make a new one.
        puzzle_dict = {}
    if override:
        try:
            puzzle_dict.pop(day)
        except:
            pass
    try:
        puzzle_input = puzzle_dict[day]
    except:
        puzzle_input = _pull_puzzle_input(day, seperator, cast)
        puzzle_dict[day] = puzzle_input
        pickle.dump(puzzle_dict, open(_code_path + r'\{}\input.p'.format(_year), 'wb'))
    return puzzle_input


def _day1():
    raw_expense = get_input(1, '\n', int)
    expense_set =  set()
    # Create a set from the list to make lookups faster
    for expense in raw_expense:
        expense_set.add(expense)
    for expense in expense_set:
        temp = 2020 - expense
        if temp in expense_set:
            print("Part 1")
            print(" {} + {} = {}".format(expense, temp, 2020))
            print(" {} * {} = {}".format(expense, temp, temp * expense))
            break
    expense_arr = numpy.array(raw_expense, dtype=int, copy=True)
    initial_size = len(expense_arr)
    # Take the two smallest values and cull anything too big for this solution
    smallest = numpy.partition(expense_arr, 0)[0]
    biggest = 2020 - (smallest + numpy.partition(expense_arr, 1)[1])
    expense_arr = expense_arr[expense_arr <= biggest]
    found = False
    print("Part 2")
    print(" Removed {} expenses from the search".format(initial_size - len(expense_arr)))
    counter = 0
    for expense in expense_arr:
        # Filter the array for values that would fit this expense + the smallest value
        remaining_arr = expense_arr[expense_arr <= (2020 - (expense + smallest))]
        if len(remaining_arr) < 2:
            continue
        # Search this remaining array
        for remaining_expense in remaining_arr:
            counter += 1
            temp = 2020 - remaining_expense - expense
            if temp in remaining_arr:
                print(" {} + {} + {} = 2020".format(expense, remaining_expense, temp))
                print(" {} * {} * {} = {}".format(expense, remaining_expense, temp, (expense * remaining_expense * temp)))
                found = True
                break
        if found:
            break
    print(" Search loop counter = {}".format(counter))


def _day2():
        puzzle = get_input(2, '\n', None)
        valid_count = 0
        valid2_count = 0
        for data in puzzle:
            part2_counter = 0
            # Data formatting
            rule, password = data.split(": ")  # Rule and the password are separated by :
            count, char = rule.split(" ")  # a count (or position) in the format 1-2 are separated from the character by ' '
            minimum, maximum = map(int, count.split("-"))
            # Part 1 test
            if password.count(char) in range(minimum, maximum + 1):
                valid_count += 1
            # Part 2 test minimum and maximum are now 1 based index positions into the password.
            if password[minimum - 1] == char:
                part2_counter += 1
            if password[maximum - 1] == char:
                part2_counter += 1
            if part2_counter == 1:
                valid2_count += 1
        print("Part 1")
        print(" Valid password count: {}".format(valid_count))
        print("Part 2")
        print(" Valid password count: {}".format(valid2_count))


def _day3():
    puzzle = get_input(3, '\n', list)
    width = len(puzzle[0])
    height = len(puzzle)
    right = 3
    down = 1
    current_pos = [0,0]
    #              y,x
    tree_counter = 0
    while current_pos[0] < height:
        if puzzle[current_pos[0]][current_pos[1]] == "#":
            tree_counter += 1
        current_pos[0] += down
        current_pos[1] += right
        current_pos[1] = current_pos[1] % width
    print("Part 1")
    print(" Tree count: {}".format(tree_counter))
    directions = [[1,1],[3,1],[5,1],[7,1],[1,2]]
    tree_count = []
    for right, down in directions:
        current_pos = [0,0]
        #              y,x
        tree_counter = 0
        while current_pos[0] < height:
            if puzzle[current_pos[0]][current_pos[1]] == "#":
                tree_counter += 1
            current_pos[0] += down
            current_pos[1] += right
            current_pos[1] = current_pos[1] % width        
        tree_count.append(tree_counter)
    print("Part2")
    print(" Trees: {}".format(tree_count))
    print(" Multiplied: {}".format(reduce((lambda x, y: x * y), tree_count)))


def _day4():
    puzzle = get_input(4, '\n', None)
    passport_list = []
    this_passport = {}
    puzzle = puzzle + ("",)  # Need a newline to terminate the last passport in the list.
    for line in puzzle:
        if line is "" and this_passport is not {}:
            passport_list.append(this_passport)
            this_passport = {}
        data_list = line.split()
        for data in data_list:
            key, value = data.split(":")
            this_passport[key]=value
    test_set = set(["byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid"])
    counter = 0
    for passport in passport_list:
        if test_set.issubset(set(passport.keys())):
            counter += 1
    print("Part 1")
    print(" Valid passports: {}".format(counter))
    counter = 0
    for passport in passport_list:
        if test_set.issubset(set(passport.keys())):
            if int(passport["byr"]) not in range(1920, 2003):
                continue
            if int(passport["iyr"]) not in range(2010, 2021):
                continue
            if int(passport["eyr"]) not in range(2020, 2031):
                continue
            if "cm" not in passport["hgt"] and "in" not in passport["hgt"]:
                continue
            if "cm" in passport["hgt"] and int(passport["hgt"].rstrip("cm")) not in range(150, 194):
                continue
            if "in" in passport["hgt"] and int(passport["hgt"].rstrip("in")) not in range(59, 77):
                continue
            test_str = "0123456789abcdef"
            if "#" not in passport["hcl"] or len(passport["hcl"]) != 7 or any(x not in test_str for x in passport["hcl"].lstrip("#")):
                continue
            if passport["ecl"] not in ["amb", "blu", "brn", "gry", "grn", "hzl", "oth"]:
                continue
            test_str = "0123456789"
            if len(passport["pid"]) != 9 or any(x not in test_str for x in passport["pid"]):
                continue
            counter += 1
    print("Part 2")
    print(" Valid passports: {}".format(counter))


def _binary_part(data):
    possible = list(range(2**len(data)))
    for choice in data:
        if choice.lower() in ["f", "l"]:  # Lower half
            possible = possible[:len(possible) // 2]
        elif choice.lower() in ["b", "r"]:  # Upper half
            possible = possible[len(possible) // 2:]
        else:
            raise Exception("Unexpected move {}".format(choice))
    if len(possible) != 1:
        raise Exception("Did not reduce to a single choice?\n{}\n{}".format(data,possible))
    return possible[0]


def _day5():
    puzzle = ["BFFFBBFRRR", "FFFBBBFRRR", "BBFFBBFRLL"]
    puzzle = get_input(5, '\n', None)
    max_id = 0
    pass_id_list = []
    for boarding_pass in puzzle:
        row = list(boarding_pass[:7])
        column = list(boarding_pass[7:])
        pass_id = (_binary_part(row) * 8) + _binary_part(column)
        pass_id_list.append(pass_id)
        if pass_id > max_id:
            max_id = pass_id
        #print("{} - {}".format(boarding_pass, pass_id))
    print("Part 1")
    print(" highest seat ID: {}".format(max_id))
    print("Part 2")
    pass_id_list = sorted(pass_id_list)
    for index in range(len(pass_id_list)):
        if index + 1 < len(pass_id_list) and pass_id_list[index + 1] - pass_id_list[index] == 2:
            print(" My seat ID: {}".format(pass_id_list[index] + 1))
            break


def _day6():
    # Puzzle example
    # puzzle = tuple(["abc","","a","b","c","","ab","ac","","a","a","a","a","","b",""])
    # puzzle = list(map(set, puzzle))
    puzzle = get_input(6, '\n', set)
    # Add a new line at the end to terminate the last group of answers
    group_list = [list(y) for x, y in itertools.groupby(puzzle, lambda z: z == set()) if not x]
    print("Part 1 - sum of unique yes answers: {}".format(sum(map(len, [set.union(*x) for x in group_list]))))
    print("Part 2 - sum of matching yes answers: {}".format(sum(map(len, [set.intersection(*x) for x in group_list]))))


def _gold_bag_search(bag_dict, key):
    if bag_dict[key] == {}:
        return False
    if "shiny gold bag" in bag_dict[key].keys():
        return True
    for interior_bag in bag_dict[key].keys():
        if _gold_bag_search(bag_dict, interior_bag):
            return True
            break
    return False


def _gold_bag_count(bag_dict, key):
    counter = 1
    if bag_dict[key] != {}:
        for interior_bag in bag_dict[key].keys():
            counter += (bag_dict[key][interior_bag] * _gold_bag_count(bag_dict, interior_bag))
    return counter


def _day7():
    example1 = ["light red bags contain 1 bright white bag, 2 muted yellow bags.",
                "dark orange bags contain 3 bright white bags, 4 muted yellow bags.",
                "bright white bags contain 1 shiny gold bag.",
                "muted yellow bags contain 2 shiny gold bags, 9 faded blue bags.",
                "shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.",
                "dark olive bags contain 3 faded blue bags, 4 dotted black bags.",
                "vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.",
                "faded blue bags contain no other bags.",
                "dotted black bags contain no other bags."
               ]
    example2 = ["shiny gold bags contain 2 dark red bags.",
                "dark red bags contain 2 dark orange bags.",
                "dark orange bags contain 2 dark yellow bags.",
                "dark yellow bags contain 2 dark green bags.",
                "dark green bags contain 2 dark blue bags.",
                "dark blue bags contain 2 dark violet bags.",
                "dark violet bags contain no other bags."
               ]
    puzzle = get_input(7, '\n', None)
    # puzzle  = example1
    # puzzle = example2
    bag_dict = {}
    # Rule processing block, dictionary contains bag name as the key and # of that bag as a value.
    # i.e. {"light red bag": {"bright white bag": 1, "muted yellow bag": 2}}
    for rule in puzzle:
        outside, inside = rule.split(" contain ")
        outside = outside.rstrip("s")  # Make all the key names end with bag (no bags)
        bag_dict.setdefault(outside, {})
        if "no other bags" not in inside:
            for bags in inside.split(","):
                number = int(bags.split()[0])
                bag_type = bags.lstrip("{} ".format(number)).rstrip(".").rstrip("s")
                bag_dict[outside][bag_type.rstrip(".")] = int(number)
    contains_gold = 0
    # Lazy recursive searching...
    for bag in bag_dict:
        if _gold_bag_search(bag_dict, bag):
            contains_gold += 1
    print("Part 1 - {} bags contain at least one shiny gold bag".format(contains_gold))
    print("Part 2 - {} bags required for a gold bag".format(_gold_bag_count(bag_dict, "shiny gold bag") - 1))


def _program(puzzle):
    accumulator = 0
    ip_set = set()
    ip = 0
    while True:
        # print(puzzle[ip])
        if ip in ip_set:  # Infinate loop detected
            return (False, accumulator)
        elif ip == len(puzzle):  # Correct termination.
            return (True, accumulator)
        else:
            ip_set.add(ip)
        instruction, value = puzzle[ip].split()
        if instruction == "nop":
            ip += 1
        elif instruction == "acc":
            accumulator += int(value)
            ip += 1
        elif instruction == "jmp":
            ip += int(value)

def _day8():
    example1 = ["nop +0",
                "acc +1",
                "jmp +4",
                "acc +3",
                "jmp -3",
                "acc -99",
                "acc +1",
                "jmp -4",
                "acc +6"]
    puzzle = get_input(8, '\n', None)
    # puzzle = example1
    print("Part 1 - accumulator was {}".format(_program(puzzle)[1]))
    for index in range(len(puzzle)):
        instruction, value = puzzle[index].split()
        if instruction == "acc":
            continue
        modified_puzzle = list(puzzle)
        if instruction == "jmp":
            modified_puzzle[index] = modified_puzzle[index].replace("jmp", "nop")
        if instruction == "nop":
            modified_puzzle[index] = modified_puzzle[index].replace("nop", "jmp")
        # print("testing index {}".format(index))
        # print(modified_puzzle)
        result, accumulator = _program(modified_puzzle)
        if result:
            print("Part 2 - accumulator was {}".format(accumulator))
            break


def _day9():
    puzzle = tuple([35,20,15,25,47,40,62,55,65,95,102,117,150,182,127,219,299,277,309,576])
    preamble_count = 5
    puzzle = get_input(9, '\n', int)
    preamble_count = 25
    for index in range(preamble_count, len(puzzle)):
        preamble = numpy.array(puzzle[index - preamble_count : index])
        subtracted = (preamble - puzzle[index]) * - 1
        intersection = numpy.intersect1d(preamble, subtracted)
        if len(intersection) == 0:
            invalid_number = puzzle[index]
            print("First failing value: {}\n  preamble:     {}\n  remainder:    {}\n  intersection: {}\n\n".format(puzzle[index], preamble, subtracted, intersection))
            break
    # Can't think of a super elegent solution right now
    # Time for some brute force searching of all the possible contiguous number sets.
    for contiguous in range(2, len(puzzle)):
        for index in range(0, len(puzzle) - contiguous):
            if sum(puzzle[index:index+contiguous]) == invalid_number:
                answer = min(puzzle[index:index+contiguous]) + max(puzzle[index:index+contiguous])
                print("Answer: {}\n{}".format(answer, puzzle[index:index+contiguous]))
                break

def _walk_forward(counter, index, bag):
    joltage = 0 if index == 0 else bag[index-1]
    if index == len(bag):
        counter += 1
    else:
        for i in range(3):
            if index+i >= len(bag):
                break
            if bag[index+i] - 3 <= joltage:
                #print("c:{} {}".format(counter, bag[index+i]))
                counter = _walk(counter, index+i+1, bag)
    return counter


def _day10():
    #bag = [1,4,5,6,7]
    #bag = [16,10,15,5,1,11,7,19,6,12,4]
    bag = [28,33,18,42,31,14,46,20,48,47,24,23,49,45,19,38,39,11,1,32,25,35,8,17,7,9,4,2,34,10,3]
    #bag = [1,3,4,5]
    bag = list(get_input(10, '\n', int))
    bag = sorted(bag)
    bag.append(max(bag)+3)
    # Ok this was half experimenting and half cheating.
    # Reddit mentioned tribonacci sequence. I used the recursive function to check
    # That in a group of adapters which increment by 1 the permutations follows the sequence.
    # The groups end up <4 in the puzzle.
    # Hard coding the values but this could either use the recursion function or a direct tribonacci calc
    trib_lookup = {0:1, 1:1, 2:2, 3:4, 4:7, 5:13}
    joltage = 0
    jolt_list = [0,0,0,0]
    jolt_group_dict = {}
    ones_counter = 0
    for adapter in bag:
        difference = adapter - joltage
        if difference == 1:  # Continued group of adapters which each is +1 the previous.
            ones_counter += 1
        elif difference == 3: # Boundary for the group of +1 adapters.
            jolt_group_dict.setdefault(ones_counter, 0)
            jolt_group_dict[ones_counter] += 1
            ones_counter = 0
        else:
            raise Exception("Difference was {}-{}={}!".format(adapter, joltage, difference))
        if difference > 3:
            raise Exception("Error, joltage={} adapter={}".format(joltage, adapter))
        jolt_list[difference] += 1
        joltage = adapter
    print("Part 1")
    print(" Jolt different list: {}".format(jolt_list))
    print(" Answer: {}".format(jolt_list[1]*jolt_list[3]))
    # Noticed that an adapter to adapter difference of 3 is a single point (no permutations)
    # This ends up as a divider between the other sections that have permutations.
    # Can get the total permutations by multiplying permutations of all the groups together.
    print("Part 2")
    #print(jolt_group_dict)
    combinations = 1
    for group, occurance in jolt_group_dict.items():
        # Each block of adapters that increment by 1 have valid permutations which follow the tribonacci sequence.
        # Multiple them together
        combinations = combinations * (trib_lookup[group] ** occurance)
    print(" Answer: {}".format(combinations))



def _step_adjacent(first, second):
    for y in range(len(first)):
        for x in range(len(first[0])):
            left = 0 if x-1 < 0 else x-1
            right = x if x+1 == len(first[0]) else x+1
            up = 0 if y-1 < 0 else y-1
            down = y if y+1 == len(first) else y+1
            if first[y][x] == 1 and numpy.count_nonzero(first[up:down+1,left:right+1] == 3) == 0:
                second[y][x] = 3
            elif first[y][x] == 3 and numpy.count_nonzero(first[up:down+1,left:right+1] == 3) > 4:
                #print("Empty the seat at {},{} [{}:{},{}:{}]".format(x,y,up,down,left,right))
                #print(first[up:down+1,left:right+1])
                second[y][x] = 1

def _test(array):
    if next((x for x in array if x != 0), 0) == 3:
        return 1
    else:
        return 0

def _step_diagonals(first, second):
    """
    Diagonal view points... again
    """
    for y in range(len(first)):
        for x in range(len(first[0])):
            # [up, down,
            #  left, right,
            #  down-right, up-left
            #  down-left, up-right]
            up = numpy.flip(first[:y, x:x+1].T)[0]
            down = first[y+1:, x:x+1].T[0]
            left = numpy.flip(first[y:y+1, :x])
            right = first[y:y+1, x+1:]
            # Down & Right
            size = min(len(first)-y, len(first[0])-x)
            temp = first[y+1:size+y,x+1:size+x]
            down_right = temp if numpy.size(temp) < 4 else numpy.diagonal(temp)
            # Up & Left
            size = min(y, x)
            temp = first[y-size:y,x-size:x]
            up_left = temp if numpy.size(temp) < 4 else numpy.flip(numpy.diagonal(temp))
            # Up & Right
            size = min(y, len(first[0])-x-1)
            temp = first[y-size:y,x+1:size+x+1]
            up_right = temp if numpy.size(temp) < 4 else numpy.flipud(temp).diagonal()
            # Down & Left
            size = min(len(first[0])-y-1, x)
            temp = first[y+1:size+y+1,x-size:x]
            down_left = temp if numpy.size(temp) < 4 else numpy.fliplr(temp).diagonal()

            #print("{},{} [{}:{},{}:{}] s:{}".format(x,y,y+1,size+y+1,x-size,x,size))
            #print(" {}".format(temp))
            #print(" Test - {}".format(down_left))
            directions = [up, down, left, right, up_left, up_right, down_left, down_right]
            occupied = 0
            for direction in directions:
                if len(direction) == 1 and type(direction[0]) in [numpy.array, numpy.ndarray, list]:
                    occupied += _test(direction[0])
                else:
                    occupied += _test(direction)
                #if x == 5 and y == 3:
                #    print("{} - {}".format(occupied, direction))
            if first[y][x] == 1 and occupied == 0:
                second[y][x] = 3
            elif first[y][x] == 3 and occupied >= 5:
                second[y][x] = 1
            #print("{},{} [{}:{},{}:{}] s:{}".format(x,y,y+1,size+y+1,x-size,x,size))

def _check(x, y, first):
    if x < 0 or y < 0:
        return -1
    try:
        temp = first[y][x]
    except:
        temp = -1
    # Empty spaces don't count
    if temp == 0:
        temp = None
    return temp


def _box(x, y, first):
    """
    Start from x,y search outward checking the 8 corners
    """
    result = [None] * 8  # [u, ur, r, dr, d, dl, l, ul]
    size = 1
    p = False
    #if x == 5 and y == 0:
    #    p = True
    while True:
        # Up
        y -= size
        if p:
            print("s: {} x,y: {},{} up: {}".format(size, x,y, _check(x,y, first)))
        result[0] = _check(x, y, first) if result[0] is None else result[0]
        # Right = Upper right
        x += size
        result[1] = _check(x, y, first) if result[1] is None else result[1]
        # Down = Right
        y += size
        result[2] = _check(x, y, first) if result[2] is None else result[2]
        # Down = Lower right
        y += size
        result[3] = _check(x, y, first) if result[3] is None else result[3]
        # Left = Down
        x -= size
        result[4] = _check(x, y, first) if result[4] is None else result[4]
        # Left = Lower left
        x -= size
        result[5] = _check(x, y, first) if result[5] is None else result[5]
        # Up = Left
        y -= size
        result[6] = _check(x, y, first) if result[6] is None else result[6]
        # Up = Upper left
        y -= size
        result[7] = _check(x, y, first) if result[7] is None else result[7]
        # return down and right
        y += size
        x += size
        if p:
            print("result: {}".format(result))
        size += 1
        if result.count(None) == 0:
            break
    return result.count(3)


def _step_box(first, second):
    """
    Trying a box search
    """
    for y in range(len(first)):
        for x in range(len(first[0])):
            if first[y][x] == 0:
                continue
            else:
                occupied = _box(x, y, first)
                if first[y][x] == 1 and occupied == 0:
                    second[y][x] = 3
                elif first[y][x] == 3 and occupied >= 5:
                    second[y][x] = 1
                #if x == 5 and y == 0 and :
                #    print("o:{} f:{} s:{}".format(occupied, first[y][x], second[y][x]))


def _decode(s):
    s_list = list(s)
    for i in range(len(s_list)):
        if s_list[i] == "L":
            s_list[i] = 1
        else:
            s_list[i] = 0
    return s_list

def _print_f(ferry):
    for y in range(len(ferry)):
        s=str(ferry[y])
        s=s.replace('[','')
        s=s.replace(']','')
        s=s.replace(',','')
        s=s.replace('0','.')
        s=s.replace('1','L')
        s=s.replace('3','#')
        s=s.replace(' ','')
        print(s)    


def _day11():
    """
    floor = 0
    empty = 1
    Occupied = 3
    """
    debug_print = False
    puzzle = ["L.LL.LL.LL",
            "LLLLLLL.LL",
            "L.L.L..L..",
            "LLLL.LL.LL",
            "L.LL.LL.LL",
            "L.LLLLL.LL",
            "..L.L.....",
            "LLLLLLLLLL",
            "L.LLLLLL.L",
            "L.LLLLL.LL"]
    puzzle = list(map(_decode, puzzle))
    puzzle = get_input(11, '\n', _decode, override=True)
    ferry = [numpy.array(puzzle),numpy.array(puzzle)]
    while True:
        ferry.reverse()
        ferry[1] = numpy.copy(ferry[0])
        _step_adjacent(*ferry)
        if debug_print:
            print("Before")
            print(ferry[0])
            print("After")
            print(ferry[1])
            print("########################")
        if numpy.array_equal(*ferry):
            break
    print("Part 1 - occupied seats {}".format(numpy.count_nonzero(ferry[0] == 3)))
    ferry = [numpy.array(puzzle),numpy.array(puzzle)]
    while True:
        ferry.reverse()
        ferry[1] = numpy.copy(ferry[0])
        _step_box(*ferry)  # Well, slicing didn't work so....
        if debug_print:
            print("Before")
            _print_f(ferry[0])
            print("After")
            _print_f(ferry[1])
            print("-"*len(ferry[0][0]))
        if numpy.array_equal(*ferry):
            break
    print("Part 2 - occupied seats {}".format(numpy.count_nonzero(ferry[0] == 3)))


def _day12():
    puzzle = ["F10","N3","F7","R90","F11"]
    puzzle = get_input(12, "\n", None)
    ordinals = {
        "N": numpy.array([0,1]),
        "S": numpy.array([0,-1]),
        "E": numpy.array([1,0]),
        "W": numpy.array([-1,0])}
    right = {"N":"E", "E":"S", "S":"W", "W":"N"}
    left  = {"N":"W", "W":"S", "S":"E", "E":"N"}
    cur_dir = "E"
    cur_pos = numpy.array([0,0]) # X, Y
    for direction in puzzle:
        if direction[:1] in ordinals.keys():
            cur_pos = cur_pos + (ordinals[direction[:1]] * int(direction[1:]))
        elif "F" in direction:
            cur_pos = cur_pos + (ordinals[cur_dir]       * int(direction[1:]))
        elif "R" in direction:
            for i in range(int(direction[1:])//90):
                cur_dir = right[cur_dir]
        elif "L" in direction:
            for i in range(int(direction[1:])//90):
                cur_dir = left[cur_dir]
        else:
            raise Exception("Unsupported direction {}".format(direction))
    print("Part 1")
    print(" cur_pos: {} = {}".format(cur_pos, sum(map(abs, cur_pos))))
    cur_pos = numpy.array([0,0])
    way_point = numpy.array([10,1])
    for direction in puzzle:
        if direction[:1] in ordinals.keys():
            way_point = way_point + (ordinals[direction[:1]] * int(direction[1:]))
        elif "F" in direction:
            cur_pos = cur_pos + (way_point * int(direction[1:]))
        elif "R" in direction:
            for i in range(int(direction[1:]) // 90):
                way_point = numpy.flip(way_point)
                way_point = way_point * numpy.array([1,-1])
        elif "L" in direction:
            for i in range(int(direction[1:]) // 90):
                way_point = numpy.flip(way_point)
                way_point = way_point * numpy.array([-1,1])
        else:
            raise Exception("Unsupported direction {}".format(direction))
        #print("current {} waypoint {}".format(cur_pos, way_point))
    print("Part 2")
    print(" cur_pos: {} = {}".format(cur_pos, sum(map(abs, cur_pos))))
        

def _day13():
    puzzle = ["939","7,13,x,x,59,x,31,19"]
    #puzzle = ["1","1789,37,47,1889"]
    #puzzle = get_input(13, '\n', None)
    earliest = int(puzzle[0])
    bus_set = set()
    for data in puzzle[1].split(','):
        if data != "x":
            bus_set.add(int(data))

    # Lazy walk forward
    """
    time = earliest - 1
    found = False
    while found is False:
        time += 1
        for bus in bus_set:
            if time % bus == 0:
                #print("Bus {} at time {}".format(bus, time))
                found = True
                break
    print("Part 1")
    print(" Bus {} at time {}".format(bus, time))
    print(" answer {}".format( (time-earliest) * bus))
    """
    # This probably wont work.
    time=0
    found = False
    bus_list = puzzle[1].split(',')
    min_step = min(bus_set)
    max_interval = 0
    for i in range(len(bus_list)):
        if bus_list[i] == "x":
            continue
        if int(bus_list[i]) + i > max_interval:
            max_interval = int(bus_list[i]) + i
    print("max {}".format(max_interval))
    return
    while found is False:
        time += max_interval
        print(time)
        found = True
        for i in range(len(bus_list)):
            if bus_list[i] == "x":
                continue
            if (time + i) % int(bus_list[i]) != 0:
                found = False
                break
    print("time {}".format(time))


def _floating(address, mask):
    """
    Recusive function to handle that X is both 0 and 1.
    """
    if "X" not in mask:  # No new address generated
        return set([address])
    else:
        temp = mask[::-1]  # Reverse the mask so that the python index values match the bit position (msb was on the left in the mask)
        i = temp.index("X")  # Find the first X
        temp = temp[:i] + "0" + temp[i+1:]  # Make a new mask with this bit removed.
        new_mask = temp[::-1]  # Reverse again?
        # Find the two new addresses recursively....
        address_set = _floating((address | (1 << i)), new_mask)
        address_set = address_set.union(_floating( (address & (~(1 << i) & (2**36-1))), new_mask))
        return address_set


def _day14():
    puzzle = ["mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X",
              "mem[8] = 11",
              "mem[7] = 101",
              "mem[8] = 0"]
    puzzle = ["mask = 000000000000000000000000000000X1001X",
              "mem[42] = 100",
              "mask = 00000000000000000000000000000000X0XX",
              "mem[26] = 1"]
    puzzle = get_input(14, '\n', None)
    and_mask = (2**36) - 1
    or_mask = 0
    float_mask = "0"
    p1_memory = {}
    p2_memory = {}
    for line in puzzle:
        command, value = line.split(" = ")
        if command == "mask":
            and_mask = int(value.replace("X", "1"), 2)
            or_mask = int(value.replace("X", "0"), 2)
            float_mask = value
        elif "mem" in command:
            address = int(command.split("]")[0].split("[")[1])
            value = int(value, 10)
            p1_memory[address] = (value & and_mask) | or_mask  # clear and set bits according to the mask for part 1.
            for an_address in _floating(address, float_mask):  # Write to all the addresses according to the mask for part 2.
                p2_memory[an_address | or_mask] = value
        else:
            raise Exception("Unknown command {}".format(line))
    print("Part 1 - {}".format(sum(p1_memory.values())))
    print("Part 2 - {}".format(sum(p2_memory.values())))


def _day15():
    starting_num = [0,3,6]
    #starting_num = [1,3,2]
    #starting_num = [3,1,2]
    #starting_num = [1,0,15,2,10,13]
    tracking_dict = {}
    # Load the inital set
    last_number = starting_num.pop()
    for index in range(len(starting_num)):
        number = starting_num[index]
        if number in tracking_dict:
            raise Exception("Hmmm")
        else:
            tracking_dict[number] = index + 1
    turn = len(starting_num)+1
    # Start the game.
    while turn < 30000000:
        if turn == 2020:
            print("On turn {} the number was {}".format(turn, last_number))
        #print("t:{} ln:{} td:{}".format(turn, last_number, tracking_dict))
        if last_number in tracking_dict.keys():
            prev = tracking_dict[last_number]
            tracking_dict[last_number] = turn
            last_number = turn - prev
        else:
            tracking_dict[last_number] = turn
            last_number = 0
        turn += 1
    print("On turn {} the number was {}".format(turn, last_number))


# Trying to reduce the speed by using a list instead of a dictionary
def _day15_a():
    starting_num = [0,3,6]
    starting_num = [1,0,15,2,10,13]
    tracking = [0]*30000000
    start = len(starting_num)
    last_num = starting_num.pop()
    for i in range(len(starting_num)):
        tracking[starting_num[i]]=i+1
    for turn in range(start,30000000):
        if turn == 2020:
            print("On turn {} the number was {}".format(turn, last_num))
        prev = tracking[last_num]
        tracking[last_num] = turn
        last_num = 0 if prev == 0 else turn - prev
    print("On turn 30000000 the number was {}".format(last_num))


def _day16():
    puzzle = [
        "class: 1-3 or 5-7",
        "row: 6-11 or 33-44",
        "seat: 13-40 or 45-50",
        "",
        "your ticket:",
        "7,1,14",
        "",
        "nearby tickets:",
        "7,3,47",
        "40,4,50",
        "55,2,20",
        "38,6,12"]
    puzzle = [
        "class: 0-1 or 4-19",
        "row: 0-5 or 8-19",
        "seat: 0-13 or 16-19",
        "",
        "your ticket:",
        "11,12,13",
        "",
        "nearby tickets:",
        "3,9,18",
        "15,1,5",
        "5,14,9 "]
    puzzle = [
        "departure location: 41-526 or 547-973",
        #"departure station: 29-874 or 891-961",
        #"departure platform: 25-200 or 213-966",
        #"departure track: 38-131 or 152-951",
        #"departure date: 29-349 or 366-955",
        #"departure time: 34-450 or 464-958",
        "",
        "your ticket:",
        "197,173,229,179,157,83,89,79,193,53,163,59,227,131,199,223,61,181,167,191",
        "",
        "nearby tickets:",
        "153,109,923,689,426,793,483,628,843,774,785,841,63,168,314,725,489,339,231,914",
        "177,714,226,83,177,199,186,227,474,942,978,440,905,346,788,700,346,247,925,825",
        "98,718,599,348,225,261,310,490,773,867,659,874,286,290,408,481,780,240,309,391"]
    puzzle = get_input(16, "\n", None, True)
    your_ticket = False
    nearby_tickets = False
    test_set = set()
    nearby_dict = {}
    invalids = 0
    test_dict = {}
    for line in puzzle:
        if line == "":
            continue
        if "your ticket" in line:
            your_ticket = True
            continue
        if "nearby tickets" in line:
            nearby_tickets = True
            continue
        if not your_ticket and not nearby_tickets:
            #print(line)
            field, ranges = line.split(": ")
            ranges_list = ranges.split(" or ")
            test_dict.setdefault(field, set())
            for r in ranges_list:
                start, end = map(int, r.split("-"))
                this_set = set(range(start, end+1))
                test_dict[field] = test_dict[field].union(this_set)
                test_set = test_set.union(set(range(start, end+1)))
        elif your_ticket and not nearby_tickets:
            my_ticket = map(int, line.split(","))
        elif nearby_tickets:
            nearby = list(map(int, line.split(",")))
            if len(nearby) != 20:
                print("?!?")
            invalids += sum(set(nearby) - test_set)
            for i in range(len(nearby)):
                nearby_dict.setdefault(i, set())
                nearby_dict[i].add(nearby[i])
        else:
            raise Exception("Unhandled line {}".format(line))
    print("Part 1: {}".format(invalids))
    """
    for k, v in test_dict.items():
        print(k)
        print(v)
        print("")
    for p, v in nearby_dict.items():
        print("Position {}".format(p))
        print(v)
        """
    #print("test_dict: {}".format(test_dict))
    #print("nearby_dict[0]: {}".format(sorted(nearby_dict[0])))
    for rule, rule_set in test_dict.items():
        if "departure" not in rule:
            continue
        print("Testing {}".format(rule))
        #print("Testing {}\n{}".format(rule, rule_set))
        for pos, value_set in nearby_dict.items():
           # print(" Position {}\n {}".format(pos, value_set))
            if value_set.issubset(rule_set):
                print("Position {} matches rule {}".format(pos, rule))
        

        


def go(day):
    switch = {
        1: _day1,
        2: _day2,
        3: _day3,
        4: _day4,
        5: _day5,
        6: _day6,
        7: _day7,
        8: _day8,
        9: _day9,
        10: _day10,
        11: _day11,
        12: _day12,
        13: _day13,
        14: _day14,
        15:_day15_a,
        16:_day16,
    }
    return switch.get(day, "Invalid day")()
