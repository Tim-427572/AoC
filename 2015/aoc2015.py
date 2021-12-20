import requests
import numpy
import pickle
from os import path
from functools import reduce
import itertools
import random

# The Value from the session cookie used to make the webaccess.
# You could hardcode this with your value or set it at the interactive prompt.
# This is because I am lazy and didn't want to figure out how to scrape the cookie or work with the OAuth.
# I'd never work on these at the office but...
_code_path = r'c:\AoC'
_work = True
_offline = False
_year = 2015

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
            raise Exception("Using the web browser get the session cookie value\nand put it as a string in {}".format(_code_path + "\session.txt"))  # noqa: W605
        with open(_code_path + "/session.txt", 'r') as session_file:
            session = session_file.read()
        if _work:
            proxy_dict = {'http': 'proxy-dmz.intel.com:911',
                          'https': 'proxy-dmz.intel.com:912'}
        else:
            proxy_dict = {}
        header = {'Cookie': 'session={:s}'.format(session.rstrip('\n'))}
        with requests.Session() as session:
            resp = session.get('https://adventofcode.com/{}/day/{}/input'.format(_year, day), headers = header, proxies = proxy_dict)
            text = resp.text.strip("\n")
            if resp.ok:
                if seperator is None:
                    return resp.text
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
    directions = get_input(1)
    print("Santa should go to floor {}".format(directions.count("(") - directions.count(")")))
    for i in range(len(directions)):
        if directions[:i].count("(") - directions[:i].count(")") == -1:
            print("First enter basement at position {}".format(i))
            break


def _day2_split(dimension_str):
    """
    Take the dimensions and return a sorted list
    """
    return sorted(map(int,dimension_str.split("x")))


def _day2(override=False):
    packages = get_input(2, '\n', _day2_split, override)
    area = 0
    ribbon = 0
    for package in packages:
        area += 3*package[0]*package[1] + 2*package[1]*package[2] + 2*package[2]*package[0]
        ribbon += (2 * sum(package[:2]) + np.prod(package))
    print("The elves need {} sqft. of wrapping paper".format(area))
    print("The elves need {} ft. of ribbon".format(ribbon))

def _day3(override=False):
    directions = _get_input(3, override=override)
    move_dict = {
        "^": npa(( 0, 1)),
        ">": npa(( 1, 0)),
        "<": npa((-1, 0)),
        "v": npa(( 0,-1))
        }
    santa_pos = npa((0,0))
    house_set = set()
    for move in directions:
        house_set.add(santa_pos.tostring())
        santa_pos += move_dict[move]
    print("{} houses received at least one package from Santa".format(len(house_set)))
    santa_pos = npa((0,0))
    robot_pos = npa((0,0))
    house_set = set()
    counter = 0
    for move in directions:
        house_set.add(santa_pos.tostring())
        house_set.add(robot_pos.tostring())
        if counter % 2 == 0:
            santa_pos += move_dict[move]
        else:
            robot_pos += move_dict[move]
        counter += 1
    print("{} houses received at least one package from Santa or the Robot".format(len(house_set)))


def _day4_thread(puzzle, check, range):
    import hashlib
    for i in range:
        digest = hashlib.md5((puzzle + str(i)).encode('utf-8')).hexdigest()
        if digest[:len(check)] == check:
                print(" Given {} and {} the value could be {}".format(puzzle, check, i))
                return


def _day4(check='00000', nmax=7, override=False):
    from concurrent import futures
    puzzle = "iwrupvqb"
    threads = 6
    with futures.ProcessPoolExecutor(max_workers=threads) as executor:
        running = []
        for thread in range(threads):
            r = range(1+thread, 10**nmax+thread, threads)
            f = executor.submit(_day4_thread, puzzle, check, r)
            running.append(f)
        futures.wait(running, return_when=futures.ALL_COMPLETED)


def _day5():
    import re
    strings = get_input(5, '\n', None)
    nice = 0
    for s in strings:
        nice_check = 1
        if sum(s.count(x) for x in ['ab','cd','pq','xy']):
            nice_check = 0
        if nice_check and sum(s.count(x) for x in ['a','e','i','o','u']) < 3:
            nice_check = 0
        if nice_check:
            nice_check = 0
            for i in range(len(s)-1):
                if s[i] == s[i+1]:
                    nice_check = 1
                    break
        nice += nice_check
    print("Part 1: {} strings are nice".format(nice))
    nice = 0
    for s in strings:
        nice_check = 0
        for i in range(len(s)-2):
            if s[i] == s[i+2]:
                nice_check = 1
                break
        if nice_check:
            pair = ''
            nice_check = 0
            for i in range(len(s)-1):
                pair = s[i:i+2]
                # Replace the pair with some dummy characters.
                test_str = s[:i]+"00"+s[i+2:]
                # Use regex to check for a match.
                if re.search(pair, test_str):
                    nice_check = 1
                    break
        #if nice_check:
        #    print("{} {}".format(pair,s))
        nice += nice_check
    print("Part 2: {} strings are nice".format(nice))


def _day6():
    commands = get_input(6, '\n', None)
    #commands = ["turn on 0,0 through 0,0","toggle 0,0 through 9,9"]
    lights_on = set()
    brightness = {}
    for command_str in commands:
        if "toggle" in command_str:
            command, start, _, stop = command_str.split()
        else:
            _, command, start, _, stop = command_str.split()
        startx, starty = map(int, start.split(','))
        stopx, stopy = map(int, stop.split(','))
        for position in itertools.product(range(startx, stopx+1), range(starty, stopy+1)):
            brightness.setdefault(position, 0)
            if command == "on":
                lights_on.add(position)
                brightness[position] = brightness[position] + 1
            if command == "off":
                lights_on.discard(position)
                brightness[position] = max(brightness[position] - 1, 0)
            if command == "toggle":
                brightness[position] = brightness[position] + 2
                if position in lights_on:
                    lights_on.discard(position)
                else:
                    lights_on.add(position)
    print("Part1")
    print(" {} lights are on".format(len(lights_on)))
    total_brightness = 0
    for position, level in brightness.items():
        total_brightness += level
    print("Part 2")
    print(" Total brightness is {}".format(total_brightness))


def _day7():
    connections = [
    '123 -> x',
    '456 -> y',
    'x AND y -> d',
    'x OR y -> e',
    'x LSHIFT 2 -> f',
    'y RSHIFT 2 -> g',
    'NOT x -> h',
    'NOT y -> i']
    connections = get_input(7, '\n', None)
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
                print("error")
                print(connection)
            # Make the connection (if possible)
            if source1.isnumeric():  # external signal input to wire.
                if destination == "b":
                    source1 = 46065
                wire_values[destination] = int(source1)
                remaining_connections.remove(connection)
            if source1 in wire_values.keys():  # connection to an existing wire.
                if source2 is None:  # NOT or regular connection
                    if connection_type == "NOT":
                        wire_values[destination] = wire_values[source1] ^ 0xFFFF
                    else:  # regular connection
                        wire_values[destination] = wire_values[source1]
                    remaining_connections.remove(connection)
                elif source2 is not None:  # Joining two signals
                    if source2.isnumeric():  # external signal
                        source2 = int(source2)
                    else:  # Other wire
                        if source2 not in wire_values.keys():
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
                        raise Exception("Unsupported: {}".format(connection))
                    remaining_connections.remove(connection)
    #print("Part 1")
    print("Part 2")
    print(" wire a is: {}".format(wire_values['a']))


def _day8():
    # This should be a lot easier but I'm fighting with the Python string libraries
    santa = r'""\n"abc"\n"aaa\"aaa"\n"\x27"'  # Test input
    santa = _pull_puzzle_input(8, ' ', None)[0]
    santa = santa.replace(r'"\n"', '""')
    print("Part 1")
    code_len = sum(map(len, santa))
    print(" code length: {}".format(code_len))
    # Time to brute force count the characters 
    string_values = 0
    index = 0
    while index < len(santa):
        if santa[index] != '"':
            #print(santa[index])
            if santa[index] == '\\':
                if santa[index + 1] in ['"', '\\']:
                    index += 1
                elif santa[index + 1] == "x":
                    index += 3
                else:
                    print(santa[index-5:index+5])
                    raise Exception("index: {}".format(index))
            string_values += 1
        index += 1
    print(" string values: {}".format(string_values))
    print(" answer: {}".format(code_len - string_values))

def _day9():
    distances = get_input(9, '\n', lambda a:(a.split(" ")[0], a.split(" ")[2], int(a.split(" ")[4])))
    #distances = [("London","Dublin",464),("London","Belfast",518),("Dublin","Belfast",141)]
    city_set = set()
    distance_dict = {}
    for city_a, city_b, distance in distances:
        city_set.add(city_a)
        city_set.add(city_b)
        distance_dict.setdefault(city_a, {})
        distance_dict[city_a][city_b]=distance
        distance_dict.setdefault(city_b, {})
        distance_dict[city_b][city_a]=distance
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
        #print(f"route={route} distance={route_distance}")
    print(f"Part 1 shortest distance is {min_distance}")
    print(f"Part 2 longest distance is {max_distance}")


def _day10():
    """
    Look, say
    """
    puzzle = "3113322113"
    #puzzle = "1"
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
        #print(f"process={process}, test_char={test_char}, puzzle={puzzle}, output={output}")
        puzzle = output
        if process == 39:
            print(f"Part 1 length = {len(output)}")
    print(f"Part 2 length = {len(output)}")


def _test_password(password):
    debug_print = False
    if any(x in password for x in ['i', 'o', 'l']):
        if debug_print:
            print(f"{password} contains i,o or l")
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
                print(f"index={index} pair {password[index + 2]}{password[index + 1]} {found_match}, {first_pair}")
        if ord(password[index + 1]) - ord(password[index]) == 0:
            if first_pair is not None and first_pair != password[index]:
                found_match = True
            else:
                first_pair = password[index]
            if debug_print:
                print(f"index={index} pair {password[index + 1]}{password[index]} {found_match}, {first_pair}")
    if debug_print:
        if found_match is False:
            print(f"{password} does not contain two matching pairs")
        if found_inc is False:
            print(f"{password} does not contain incrementing sequence")
    if found_match and found_inc:
        return True
    else:
        return False


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
    for l in password_list:
        output_str += chr(l)
    return output_str



def _day11():
    """
    Corporate Policy
    """
    password = "hepxcrrq"
    password = _increment("hepxxyzz")
    while _test_password(password) is False:
        password = _increment(password)
    print(f"Part 1 new password is {password}")


def _recursive_sum(thing, current_sum=0, part2=False):
    if type(thing) is list:
        for item in thing:
            if type(item) is int:
                current_sum += item
            if type(item) in [list, dict]:
                current_sum = _recursive_sum(item, current_sum, part2)
    if type(thing) is dict:
        if part2 and ("red" in thing.keys() or "red" in thing.values()):
            return current_sum
        for key, value in thing.items():
            if type(key) is int:
                current_sum += key
            if type(value) is int:
                current_sum += value
            if type(value) in [list, dict]:
                current_sum = _recursive_sum(value, current_sum, part2)
    return current_sum


def _day12():
    import json
    data = json.loads(get_input(12, None, None))
    print(f"Part 1 sum is {_recursive_sum(data)}")
    print(f"Part 2 sum is {_recursive_sum(data, part2=True)}")


def _day13():
    happiness = [
        ["Alice","would","gain","54","happiness","units","by","sitting","next","to","Bob."],
        ["Alice","would","lose","79","happiness","units","by","sitting","next","to","Carol."],
        ["Alice","would","lose","2","happiness","units","by","sitting","next","to","David."],
        ["Bob","would","gain","83","happiness","units","by","sitting","next","to","Alice."],
        ["Bob","would","lose","7","happiness","units","by","sitting","next","to","Carol."],
        ["Bob","would","lose","63","happiness","units","by","sitting","next","to","David."],
        ["Carol","would","lose","62","happiness","units","by","sitting","next","to","Alice."],
        ["Carol","would","gain","60","happiness","units","by","sitting","next","to","Bob."],
        ["Carol","would","gain","55","happiness","units","by","sitting","next","to","David."],
        ["David","would","gain","46","happiness","units","by","sitting","next","to","Alice."],
        ["David","would","lose","7","happiness","units","by","sitting","next","to","Bob."],
        ["David","would","gain","41","happiness","units","by","sitting","next","to","Carol."]]
    happiness = get_input(13, '\n', lambda a: a.split(" "))
    decoder = {"gain":1, "lose":-1}
    happiness_dict = {"Me":{}}
    attendence_set = set()
    max_happiness = 0
    for happy_list in happiness:
        attendence_set.add(happy_list[0])
        happiness_dict.setdefault(happy_list[0], {})
        happiness_dict[happy_list[0]][happy_list[-1].rstrip(".")] = decoder[happy_list[2]] * int(happy_list[3])
        happiness_dict[happy_list[0]]["Me"] = 0
        happiness_dict["Me"][happy_list[0]] = 0
    attendence_set.add("Me")  # For Part 2 add me into the seating charts
    for placement in itertools.permutations(attendence_set):
        this_happiness = 0
        placement = list(placement)
        for index in range(-1, len(placement)-1):
            #print(f"  {placement[index]} to {placement[index + 1]}")
            this_happiness += happiness_dict[placement[index]][placement[index + 1]]
            this_happiness += happiness_dict[placement[index + 1]][placement[index]]
        #print(f"Placement {placement} = {this_happiness}")
        max_happiness = max(max_happiness, this_happiness)
    if "Me" not in attendence_set:
        print(f"Part 1 optimal happiness is {max_happiness}")
    else:
        print(f"Part 2 optimal happiness is {max_happiness}")


def _day14():
    """
    Reindeer distance.
    """
    speeds = [["Comet","can","fly","14","km/s","for","10","seconds,","but","then","must","rest","for","127","seconds."],
              ["Dancer","can","fly","16","km/s","for","11","seconds,","but","then","must","rest","for","162","seconds."]]
    time = 1000
    speeds = get_input(14, '\n', lambda a: a.split(" "))
    time = 2504
    reindeer_dict = {}
    for speed in speeds:
        reindeer_dict[speed[0]] = {"speed":int(speed[3]), "fly":int(speed[6]), "rest":int(speed[-2])}
        reindeer_dict[speed[0]]["total"] = reindeer_dict[speed[0]]["fly"] + reindeer_dict[speed[0]]["rest"]
        reindeer_dict[speed[0]]["position"]=0
        reindeer_dict[speed[0]]["points"]=0

    winner = [None, 0]
    #print(reindeer_dict)
    for deer, data in reindeer_dict.items():
        # Whole time interval distance travelled
        distance = (time // data["total"]) * data["fly"] * data["speed"]
        remaining_time = time - (data["total"] * (time // data["total"]))
        string = f" {deer} travelled {distance} in {time - remaining_time}s"
        # Remaining fractional interval calculation
        distance += data["fly"] * data["speed"] if remaining_time >= data["fly"] else remaining_time * data["speed"]
        print(string + f" then to {distance} in the remaining {remaining_time}s")
        if distance > winner[1]:
            winner = [deer, distance]
    print(f"Part 1 {winner[0]} won with {winner[1]} distance")
    print()

    max_distance = 0
    for sec in range(1,time+1):
        for deer, data in reindeer_dict.items():
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
        #print(f"After {sec} seconds")
        for deer, data in reindeer_dict.items():
            foo = 1
            #print(f" {deer} at {data['position']} with {data['points']} points")
        #foo = input()
    winner = [None, 0]
    for deer, data in reindeer_dict.items():
        print(f" {deer} has {data['points']} points")
        if data["points"] > winner[1]:
            winner = [deer, data["points"]]
    print(f"Part 2 {winner[0]} won with {winner[1]} points")


def _day15():
    foo=1

_position_dict = {"children": 0*8,
                 "cats": 1*8,
                 "samoyeds": 2*8,
                 "pomeranians": 3*8,
                 "akitas": 4*8,
                 "vizslas": 5*8,
                 "goldfish": 6*8,
                 "trees": 7*8,
                 "cars": 8*8,
                 "perfumes": 9*8}

def _mfcsam_encode(string):
    """
    Create a match/mask set for the mfcsam data
    Assumes that item encoding/quantity is < 256
    """
    global _position_dict
    match = 0
    mask = 0
    item_list = string.split(", ")
    for item in item_list:
        key, value = item.split(": ")
        match |= int(value) << _position_dict[key]
        mask |= 0xFF << _position_dict[key]
    return match, mask

def _day16():
    global _position_dict
    mfcsam_data = "children: 3, cats: 7, samoyeds: 2, pomeranians: 3, akitas: 0, vizslas: 0, goldfish: 5, trees: 3, cars: 2, perfumes: 1"
    sue_data_list = get_input(16, '\n', None)
    match, _ = _mfcsam_encode(mfcsam_data)
    for sue_data in sue_data_list:
        sue = sue_data.split(": ")[0]
        knowledge = sue_data.replace(sue + ": ", "")
        sue_match, mask = _mfcsam_encode(knowledge)
        if match & mask == sue_match:
            print(f"Part 1 {sue} is a match")
            break
    # Part 2, just going to hack in a fix.
    greater = ["cats", "trees"]
    less = ["pomeranians", "goldfish"]
    adjustment_list = greater + less
    adjustment_mask = 0
    for adjustment in adjustment_list:
        adjustment_mask |= 0xFF << _position_dict[adjustment]
    #print(f"adjustment mask = {adjustment_mask:#x}")
    for sue_data in sue_data_list:
        sue = sue_data.split(": ")[0]
        knowledge = sue_data.replace(sue + ": ", "")
        sue_match, mask = _mfcsam_encode(knowledge)
        # First check the exact values
        if match & (mask & ~adjustment_mask) == sue_match:
            #print(f"{sue} - {sue_match:#022x}, {mask:#022x}, {mask & ~adjustment_mask:#022x}")
            # Next check the special case for the things we know
            match = True
            for thing in greater:
                #print(f"{thing} = {(mask >> _position_dict[thing]) & 0xFF}")
                if (mask >> _position_dict[thing]) & 0xFF == 0xFF and (sue_match >> _position_dict[thing]) & 0xFF < (match >> _position_dict[thing]):
                    match = False
            for thing in less:
                #print(f"{thing} = {(mask >> _position_dict[thing]) & 0xFF} and {(sue_match >> _position_dict[thing]) & 0xFF} >= {(match >> _position_dict[thing])}")
                if (mask >> _position_dict[thing]) & 0xFF == 0xFF and (sue_match >> _position_dict[thing]) & 0xFF > (match >> _position_dict[thing]):
                    match = False
            if match is True:
                print(f"Part 2 {sue} is a match")
                break


def _find_combinations(containers, final, container_count):
    return [pair for pair in itertools.combinations(containers, container_count) if sum(pair) == final]


def _day17():
    """
    eggnog
    """
    containers = [20, 15, 10, 5, 5]
    total = 25
    containers = get_input(17, '\n', int)
    total = 150
    possible_combinations = 0
    for i in range(len(containers) + 1):
        possible_combinations += len(_find_combinations(containers, total, i))
    print(f"Part 1 number of possible combinations is {possible_combinations}")
    for i in range(len(containers) + 1):
        possible_combinations = _find_combinations(containers, total, i)
        if len(possible_combinations) != 0:
            print(f"Part 2 minimum is {i} containers there are {len(possible_combinations)} combinations")
            break


def _day18():
    """
    Santa's lights (game of life)
    """
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
    initial_state = get_input(18, '\n', None)
    steps = 100
    initial_list = []
    for s in initial_state:
        l = []
        for c in s:
            if c == ".":
                l.append(0)
            else:
                l.append(1)
        initial_list.append(l)
    lights = numpy.array(initial_list)
    updates = numpy.copy(lights)
    #print(lights)
    for step in range(steps):
        for row in range(lights.shape[0]):
            for col in range(lights.shape[1]):
                test_array = lights[max(row-1, 0):min(row+2, lights.shape[0]), max(col-1, 0):min(col+2, lights.shape[1])]
                if lights[row, col] == 1 and numpy.sum(test_array) not in [3,4]:  # Light was on, turns off
                    updates[row, col] = 0
                elif lights[row, col] == 0 and numpy.sum(test_array) == 3:
                    updates[row, col] = 1
                else:
                    updates[row, col] = lights[row, col]
        lights = numpy.copy(updates)
    print(f"Part 1: After {steps} steps there were {numpy.sum(lights)} lights lit")
    lights = numpy.array(initial_list)
    updates = numpy.copy(lights)
    #print(lights)
    #print()
    for step in range(steps):
        # Stuck lights
        lights[0, 0] = 1
        lights[0, lights.shape[1]-1] = 1
        lights[lights.shape[0]-1, 0] = 1
        lights[lights.shape[0]-1, lights.shape[1]-1] = 1
        for row in range(lights.shape[0]):
            for col in range(lights.shape[1]):
                test_array = lights[max(row-1, 0):min(row+2, lights.shape[0]), max(col-1, 0):min(col+2, lights.shape[1])]
                if lights[row, col] == 1 and numpy.sum(test_array) not in [3,4]:  # Light was on, turns off
                    updates[row, col] = 0
                elif lights[row, col] == 0 and numpy.sum(test_array) == 3:
                    updates[row, col] = 1
                else:
                    updates[row, col] = lights[row, col]
        lights = numpy.copy(updates)
        #print(lights)
        #foo = input()
    # Stuck lights
    lights[0, 0] = 1
    lights[0, lights.shape[1]-1] = 1
    lights[lights.shape[0]-1, 0] = 1
    lights[lights.shape[0]-1, lights.shape[1]-1] = 1
    print(f"Part 2: After {steps} steps there were {numpy.sum(lights)} lights lit")


def _day19():
    """
    HOHOHO
    """
    # Parsing the data
    puzzle = ["H => HO",
              "H => OH",
              "O => HH",
              "",
              "HOHOHO"]
    # rules = {} can't use a dictionary there are duplicate keys.
    puzzle = get_input(19, '\n', None)
    machine_in = []
    machine_out = []
    for line in puzzle:
        if line == "":
            break
        i, o = line.split(" => ")
        machine_in.append(i)
        machine_out.append(o)
    machine_zip = zip(machine_in, machine_out)
    molecule = puzzle[puzzle.index("") + 1]

    # Solve part 1
    molecule_set = set()
    for machine_input, machine_output in machine_zip:
        #print(f"Testing {machine_input} => {machine_output}")
        for i in range(len(molecule)):
            if i + len(machine_input) > len(molecule):
                continue
            #print(f" i={i} checking {molecule[i:i+len(machine_input)]}")
            if molecule[i:i+len(machine_input)] == machine_input:
                new = molecule[:i] + machine_output + molecule[i+len(machine_input):]
                #print(f" new molecule = {new}")
                molecule_set.add(new)
    print(f"Part 1 there are {len(molecule_set)} distinct molecules")
    #prev_molecule = molecule
    #machine_in.pop(5)
    #machine_out.pop(5)
    found = False
    search = 1
    print("Searching")
    while found is False:
        print(f" {search}")
        order = list(range(len(machine_in)))
        random.shuffle(order)
        steps = 0
        while len(molecule) > 0:
            modified = False
            machine_zip = zip(machine_in, machine_out)
            for machine_input, machine_output in machine_zip:
                this_round = molecule.count(machine_output)
                if this_round:
                    modified = True
                steps += this_round
                molecule = molecule.replace(machine_output, machine_input)
                #print(f" Performing {machine_output} => {machine_input} {this_round} times transformation count {steps}")
                #print(molecule)
                #print()
            # CaCa phase:
            #print("CaCa => Ca")
            #for i in range(0, len(molecule), 4):
            #    if molecule[i:i+4] == "CaCa":
            #        molecule = molecule[:i] + molecule[i+2:]
            #print(molecule)
            #print()
            if molecule == "e":
                found = True
                break
            if modified is False:
                break
        search += 1


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
        if house % test_value == 0:
            if test_value * 50 >= house:
                presents += test_value
        test_value -= 1
    return presents * 11


def _day20():
    lots_of_houses = 1000000
    goal = 29000000
    # Numpy arrays of lots of houses.
    part1_houses = numpy.zeros(lots_of_houses)
    part2_houses = numpy.zeros(lots_of_houses)
    for elf in range(1, lots_of_houses):
        # Walk array starting at elf and incrementing by elf, each time deliver elf * 10 packages.
        part1_houses[elf::elf] += (10 * elf)
        # Walk the array starting at elf and stopping after 50 houses (+1 for the slice), each time deliver 11 * elf packages.
        part2_houses[elf:(elf+1)*50:elf] += 11 * elf
    
    # Use numpy nonzero to find all the houses that have >= goal packages then grab the first one (lowest) off of the list
    print(f"There were {numpy.nonzero(part1_houses >= goal)[0].size} houses that were >= {goal}")
    print("Part 1 - {}".format(numpy.nonzero(part1_houses >= goal)[0][0]))
    print("Part 2 - {}".format(numpy.nonzero(part2_houses >= goal)[0][0]))


def _day21():
    """
    RPG Sim
    """
    weapons = {
        "Dagger":     {"cost": 8, "damage":4},
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
    enemy = {"hp":109, "damage":8, "armor":2}
    cost_of_winning_fights = []
    cost_of_losing_fights = []
    for w_name, weapon in weapons.items():
        for a_name, armor in armors.items():
            for r1_name, ring1 in rings.items():
                for r2_name, ring2 in rings.items():
                    if r1_name == r2_name and r1_name != "None":
                        continue  # Can't buy two of the same ring.
                    # print(f"Fighting with {w_name}, {a_name}, {r1_name} & {r2_name}")
                    damage = weapon["damage"] + ring1.get("damage", 0) + ring2.get("damage", 0)
                    defense = armor["armor"] + ring1.get("armor", 0) + ring2.get("armor", 0)
                    cost = weapon["cost"] + armor["cost"] + ring1.get("cost", 0) + ring2.get("cost", 0)
                    player_turns = enemy["hp"] if damage - enemy["armor"] <=0 else enemy["hp"]/(damage - enemy["armor"])
                    enemy_turns = 100 if enemy["damage"] - defense <= 0 else 100/(enemy["damage"] - defense)
                    if player_turns <= enemy_turns:
                        # print("Win!")
                        cost_of_winning_fights.append(cost)
                    else:
                        cost_of_losing_fights.append(cost)
    print(f"Part 1 - The cheapest winning fight was {min(cost_of_winning_fights)}")
    print(f"Part 2 - The most expensive losing fight was {max(cost_of_losing_fights)}")

def go(day):
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

