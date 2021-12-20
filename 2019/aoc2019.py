import requests
import numpy as np
from numpy import array as npa
import itertools
import re
import copy
import curses
import pysvtools.asciitable as asciitable
import svtools.logging.toolbox as slt
import pickle
log = slt.getLogger("aoc2019", autosplit=True)
log.setFile("aoc2019.log", dynamic=True)
log.setFileFormat('time')
log.setConsoleLevel(slt.RESULT)
log.setFileLevel(slt.INFO)
log.colorLevels(True)
log.setColor(levelname="ERROR", color=slt.FG.ired)
log.setColor(levelname="WARNING", color=slt.FG.iyellow)
log.setColor(levelname="DEBUG", color=slt.FG.imagenta)

puzzle_dict = {}


# The Value from the session cookie used to make the webaccess.
# You could hardcode this with your value or set it at the interactive prompt.
# This is because I am lazy and didn't want to figure out how to scrape the cookie or work with the OAuth.
_session = None
_session = "53616c7465645f5fde72380c43f534fd99ec9d8af38b106e44c2364a59ca683764fe04663041ba9285e33d76dd4f4d68"
# I'd never work on these at the office but...
_work = True
_offline = False

class Computer:
    import copy
    __opcode_length = {1:4, 2:4, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:2, 99:0}

    def __init__(self):
        """
        """
        self.execute = False
        self.terminate = False
        self.data_out = []
        self.data_in = []
        self._mem = []
        self._mem_size = 0
        self._mem_backup = []
        self._opcode_lookup = {}
        self._cache = {}
        self._cache_table = {}
        self._hit = 0
        self._ip = 0
        self._base = 0

    def reset(self):
        """
        Reset the IP and base, reload the memory from the backup.
        """
        self.execute = False
        self.terminate = False
        self._ip = 0
        self._base = 0
        self._mem = copy.copy(self._mem_backup)
        self._mem_size = len(self._mem)
        self._cache = {}
        self._cache_table = {}
        self._hit = 0
        self.data_in = []
        self.data_out = []

    def load_mem(self,opcodes):
        """
        Load the computer memory and make a backup
        :param opcodes: provide a list or tuple of IntCode opcodes.
        """
        self._mem = copy.copy(list(opcodes))
        self._mem_backup = copy.copy(list(opcodes))
        self._mem_size = len(self._mem)

    def write_mem(self, position, value):
        """
        Write to the IntCode internal memory.
        If the access is beyond the end of memory extend it to fit.
        """
        # Clear the cache.
        if position in self._cache_table:
            try:
                self._cache.pop(self._cache_table[position])
            except Exception:
                pass
            self._cache_table.pop(position)
        try:
            self._mem[position] = value
        except:
            # Extend memory if needed
            self._mem = self._mem + [0] * (position - self._mem_size) + [0]
            self._mem_size = len(self._mem)
            self._mem[position] = value

    def read_mem(self, position):
        """
        Read the IntCode computer internal memory.
        If the access is beyond the end of memory extend it to fit.
        """
        try:
            return self._mem[position]
        except:
            # Extend memory if needed
            self._mem = self._mem + [0] * (position - self._mem_size) + [0]
            self._mem_size = len(self._mem)
            return self._mem[position]

    def go(self):
        """
        Start the IntCode computer processing opcodes.

        execute - opcode processing in progress
        terminate - set when the opcode 99 is reached.
        """
        self.execute = True
        while(self.execute):
            # copy of the IP to monitor for any jump.
            check_ip = self._ip
            try:
                opcode, arg_pos = self._cache[self._ip]
                #self._hit += 1
            except:
                instruction = self._mem[self._ip]
                
                # Fetch the instruction decode from the lookup
                try:
                    opcode, params = self._opcode_lookup[instruction]
                except:
                    parameters, opcode = divmod(instruction, 100)
                    params = []
                    for i in range(1,self.__opcode_length[opcode]):
                        parameters, parameter = divmod(parameters, 10)
                        params.append( (parameter, i-1, i) )
                    self._opcode_lookup[instruction] = (opcode, tuple(params))
                # Decode the memory position for each argument.
                arg_pos = [-1, -1, -1]
                cacheable = True
                for parameter, index, offset in params:
                    if parameter == 2:  # Relative mode
                        # Don't cache opcodes that use relative mode.
                        cacheable = False
                        arg_pos[index] = (self._base + self.read_mem(self._ip + offset))
                    elif parameter == 1:  # Immediate mode
                        arg_pos[index] = (self._ip + offset)
                    else:  # Position mode
                        arg_pos[index] = (self.read_mem(self._ip + offset))
                if cacheable:
                    self._cache[self._ip] = (opcode, tuple(arg_pos))
                    self._cache_table[self._ip] = self._ip
                    for parameter, index, offset in params:
                        self._cache_table[self._ip + offset] = self._ip

            # Opcode processing.
            # To support access to memory beyond the initial opcode all accesses use the read_mem and write_mem functions.
            if opcode == 1:  # Add
                self.write_mem(arg_pos[2], (self.read_mem(arg_pos[0]) + self.read_mem(arg_pos[1])))
            elif opcode == 2:  # Multiple
                self.write_mem(arg_pos[2], (self.read_mem(arg_pos[0]) * self.read_mem(arg_pos[1])))
                #self._mem[self._mem[self._ip + 3]] = a + b if opcode == 1 else a * b
            elif opcode == 3:  # Input
                if len(self.data_in) == 0:
                    # No input, halt.
                    self.execute = False
                else:
                    self.write_mem(arg_pos[0], self.data_in.pop())
            elif opcode == 4:  # Output
                self.data_out.append(self.read_mem(arg_pos[0]))
            elif opcode == 5:  # Jump-if-True
                if self.read_mem(arg_pos[0]) != 0:
                    self._ip = self.read_mem(arg_pos[1])
            elif opcode == 6:  # Jump-if-False
                if self.read_mem(arg_pos[0]) == 0:
                    self._ip = self.read_mem(arg_pos[1])
            elif opcode == 7:  # less than
                if self.read_mem(arg_pos[0]) < self.read_mem(arg_pos[1]):
                    self.write_mem(arg_pos[2], 1)
                else:
                    self.write_mem(arg_pos[2], 0)
            elif opcode == 8:  # equal
                if self.read_mem(arg_pos[0]) == self.read_mem(arg_pos[1]):
                    self.write_mem(arg_pos[2], 1)
                else:
                    self.write_mem(arg_pos[2], 0)
            elif opcode == 9:  # relative base adjustment
                self._base += self.read_mem(arg_pos[0])
            elif opcode == 99:
                # Terminate
                self.execute = False
                self.terminate = True
            else:
                raise Exception("Unknown opcode:{}".format(opcode))

            # If a jump did not occur move the instruction pointer.
            if check_ip == self._ip and self.execute:
                self._ip += self.__opcode_length[opcode]


# Function to pull the data from the website
def _pull_puzzle_input(day, seperator, cast):
    """
    Pull the puzzle data from the AOC website.

    :param day: integer day value
    :param seperator: string the data seperator for the data
    :param cast: function to call on each item in the list

    :return: tuple of the data.
    """
    global _session, _work, _offline
    if _offline:
        with open(r"C:\Users\tjburbac\Downloads\day{}.txt".format(day)) as file_handler:
           data_list = file_handler.read().split(seperator)
    else:
        if _session is None:
            raise Exception("Using the web browser get the session cookie value\nand put it as a string in _session")
        if _work:
            proxy_dict = {'http': 'proxy-dmz.intel.com:911',
                          'https': 'proxy-dmz.intel.com:912'}
        else:
            proxy_dict = {}
        header = {'Cookie' : 'session={:s}'.format(_session)}
        with requests.Session() as session:
            # resp = session.get('https://topaz.github.io/paste/#XQAAAQAfOgAAAAAAAAAhlkYgxioQyWGlK76d1rYK5oF967SEBbIw3iOJC3EsYpbtDc9HMmxVbe8vTOnIJ2DRtsbYOFp+lRCe20ww64/cpt5nU2//ztz/sLAsoWJ8ziLNO0GVio/7tNYjzEYfCH090RQA2T6TR1FhG6bdTTaiTY8hp/Gxoh4ffCwnDRGrkgVOf6fSh5gZjJy2ZUDQKVOp0CJkbZOZ+dQiEkp2Ii9BV7Gj1WjDM3ZYJSnykk2sQfOsIjRB938zm4oA+VVX7l1qarbTmz3v2NHIxEGzw2793eQJ0yUm5h/YiZtx71rcmxDJO++ROMvEgttS9mZBWIWkBWVyQqB9gIp/vclAz/WXVqNBG4vgKePHlT37RT8rJQTI6KQr03y9KXzO4iHFh9WIQEXycc7gr/B1SwTXvw9OcOclZCgAioTRhcbHBuwwe3kZpCy/on53TMcxHfI5SPW/v+elFwUfVUiBqvsyYefnxrsWMJ2N4zHKp/9rPhF2ivYTcnguRCmKD8GatMdRLQ1CC0pdPPcJZXY/la4fNZx8yTW1T8AmexC4dq9oAsjrEnGiqT8s0VADesBDLiPdzS8EXjLmIMMhc8zUyR+wtYUUDD6vTY1o1BKSwhOdH6iFozh5/Qi0lF7xRGICp8D96cGi/fkh6qSqI0x7Ur90KN8UdzVPh7im6SjdzxR/0KkWULbPyyABqtHKJmkC/w8YSrbsINTwuAGiQRqmvvNBQo6MkyHYrphOdgTvcF8eVPd7MubZ4Y9tWNmDdaOipEc1b2QXe6h50KfpeP2AGW+OL2YKV99YQ50y2Wph9QCEz6leGKW8Oym4euxLLQhPNRqTEK3P9AXui09TVqZjdnUZoIFncqtIifq8qSmRg9Bity5P0VIYEm6m3d2Szp43DDhTsfbilPqwSLC08cMT3RTdSdp0iaHzXqvhs9K2a56BKbJCYNLWifNAIYizE86Fnyj6Z3HzG7+W52kCnGDXp2L17frjYaeybhWX2nbzrZKnKx+PrZR+GKLOyTTgTCtqL6y/ZUQYPHJusTtbiLcrgtj75iB4mha0o7r9H9Ai2YOr44e8EqCiSvr0LW0I+bzzZfmy3s2bITSydoUIu7C/34haQ15CvCJNXMa3Y+sxLEMj4b/QRUL1XyS+TDNwG7rXWOwSEdKOOXn4ZY0H2NH890VcAPtU/fCrpCN0dYGJr6cAzIPtbKZd/wtiUG0HK1xAzOJwukj8iMGPfK1dN4C75IuwEjoPa1PSgSeYPL6xTCIFYgB6wzPU9SA9O48aNmn83IAxOGjeIvhnXVmCQoZNEW2uIJuE7lHZfD/sBhplrHiNftlh1tgADvrKSGRkuybrQByVMxmf4Bnwa6sPQ9bJrMcFACJuTb6Z014nCx8OLnyqOo8ZhvuR6hr9BIT8JeeMOed2cykVAUDMhkjnm4pqTEmzmvgjmAthReQvHqdxaz49UpVP7o1oksb9KppMN4fc+XkxF6qYQEFsdEkAABIbpeFXtgGAPyixXO1DQHdSCdxqIdeB8EUnpTEyCTSpjN9evg0j4d+fu5//Q5QPx8KbNu4aU5Zx+hU8IxKzn9ya/JWFWxWajbD3LzLY0k3XbZEtPwRGEhmu6d+z+nNmowHShfAaX5HZnijS5zXFkI7EbNFBEwTd+lkIzUXcmHTIIglZGM+ycY3D49tWJzhqrXpUNAQeDXNjLjdGloRUVzkRHoPaQlO3edystwck3f8h1B9ZKeJNIk3lUUygab469EFCDQ3B9WnHP1FvfFQw5+ZwtGmKNMrAIJyaDmwT2Mrc8VxPJbYUNzdJJYTY3W3pFpF5BBd7lcK4OlEMMu1GKJNYZ2F4KsiHrwxfMU4FppWWZ/sYAMwc/LdcAA51qmagsvjfZX0UTGwMg2n0EaV8KLxbuQg10nnaxr0unsCz2dLEfS5CEBfy6UYssuqZVl6xJiqSuUYqmf627tGWKx7Ta'.format(day), headers = header, proxies = proxy_dict)
            resp = session.get('https://adventofcode.com/2019/day/{}/input'.format(day), headers = header, proxies = proxy_dict)
            if resp.ok:
                data_list = resp.text.split(seperator)

            else:
                print("Warning website error")
                return ()

    try:
        data_list.remove('')
    except Exception:
        foo = 1
    if cast is not None:
        data_list = [cast(x) for x in data_list]
    return tuple(data_list)

# Cache the data in a pickle file.
def get_input(day, seperator, cast, override=False):
    """
    Wrapper for the global dict.
    """
    puzzle_dict = pickle.load(open(r'C:\Users\tjburbac\Documents\AOC\aoc\input.p', 'rb'))
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
        pickle.dump(puzzle_dict, open(r'C:\Users\tjburbac\Documents\AOC\aoc\input.p', 'wb'))
    return puzzle_input


# Helper functions to unpack the puzzle data.
def _dummy(variable):
    """
    Don't change the data
    """
    return variable


def to_tup(input_line):
    """
    Day 6 unpack the orbits.
    """
    return tuple(input_line.split(")"))


# Daily puzzle solutions.

def _day1():
    mass_tup = _pull_puzzle_input(1, '\n', int)
    fuel = 0
    fuel_fuel = 0
    for mass in mass_tup:
        fuel += (mass // 3) - 2
        module_fuel = mass
        # Should have looked up making this a recursive function.
        while(True):
            module_fuel = (module_fuel // 3) - 2
            if module_fuel <= 0:
                break
            fuel_fuel += module_fuel

    print("Part 1 - Sum of the fuel requirements = {}".format(fuel))
    print("Part 2 - Sum of the fuel requirements = {}".format(fuel_fuel))


def _day2():
    #opcode_tup = _pull_puzzle_input(2, ',', int)
    opcode_tup = (1,0,0,3,1,1,2,3,1,3,4,3,1,5,0,3,2,13,1,19,1,6,19,23,2,6,23,27,1,5,27,31,2,31,9,35,1,35,5,39,1,39,5,43,1,43,10,47,2,6,47,51,1,51,5,55,2,55,6,59,1,5,59,63,2,63,6,67,1,5,67,71,1,71,6,75,2,75,10,79,1,79,5,83,2,83,6,87,1,87,5,91,2,9,91,95,1,95,6,99,2,9,99,103,2,9,103,107,1,5,107,111,1,111,5,115,1,115,13,119,1,13,119,123,2,6,123,127,1,5,127,131,1,9,131,135,1,135,9,139,2,139,6,143,1,143,5,147,2,147,6,151,1,5,151,155,2,6,155,159,1,159,2,163,1,9,163,0,99,2,0,14,0)
    # Noun, Verb for the error code was 1202
    comp = Computer()
    comp.load_mem(opcode_tup)
    comp.write_mem(1, 12)
    comp.write_mem(2, 2)
    comp.go()
    print("Part 1 - The value at position 0 is {}".format(comp.read_mem(0)))
    log.result(" Cache hit: {}".format(comp._hit))
    # Boring brute force...
    for noun in range(100):
        for verb in range(100):
            comp.reset()
            comp.write_mem(1, noun)
            comp.write_mem(2, verb)
            comp.go()
            if comp.read_mem(0) == 19690720:
                print("Part 2 - noun={}, verb={} = {}".format(noun, verb, (100*noun)+verb))
                log.result(" Cache hit: {}".format(comp._hit))
                break


def _max_dim(movement_list):
    """
    Follow the movement and return the grid size required to store the data

    :param movement_list: list of movement commands
    :return: (max positive movement [i,j], max negative movement [i,j])
    """
    cur_pos = [0,0]
    max_pos = [0,0]
    min_pos = [0,0]
    for move in movement_list:
        if "U" in move: cur_pos[0] -= int(move[1:])
        elif "D" in move: cur_pos[0] += int(move[1:])
        elif "R" in move: cur_pos[1] += int(move[1:])
        elif "L" in move: cur_pos[1] -= int(move[1:])
        else:
            raise Exception("Invalid movement command")
        max_pos[0] = max([max_pos[0], cur_pos[0]])
        max_pos[1] = max([max_pos[1], cur_pos[1]])
        min_pos[0] = min([min_pos[0], cur_pos[0]])
        min_pos[1] = min([min_pos[1], cur_pos[1]])
    return (max_pos, min_pos)


def _wire_array(movement_list, numpy_array, start_pos):
    """
    Function to map out the wire path

    :param movement_list: The movement information from the puzzle for a wire
    :param numpy_array: The array to map the movements into (by reference)
    :param start_post: list the position to use as an origin (so we don't walk off the array)

    """
    step = 1
    cur_pos = copy.copy(start_pos)
    for move in movement_list:
        direction = move[:1]
        distance = int(move[1:])
        for d in range(distance):
            if direction == "U": cur_pos[0] -= 1
            elif direction == "D": cur_pos[0] += 1
            elif direction == "R": cur_pos[1] += 1
            elif direction == "L": cur_pos[1] -= 1
            else: raise Exception("Invalid movement command")
            # Started just storing a 1 in each part of the array, switched to store the current 'step'
            #numpy_array[tuple(cur_pos)] = 1
            numpy_array[tuple(cur_pos)] = step
            step += 1

def _day3():
    # Pad the arrays just in case.
    padding = 5
    movement_tup = _pull_puzzle_input(3, '\n', _dummy)

    wire0 = movement_tup[0].split(',')
    wire1 = movement_tup[1].split(',')
    # 159 distance example
    #wire0 = ('R75','D30','R83','U83','L12','D49','R71','U7','L72')
    #wire1 = ('U62','R66','U55','R34','D71','R55','D58','R83')
    
    # Get the maximum movement infomation for both wires.
    wire0_size_tup = _max_dim(wire0)
    print("Wire0\n positive movement {}\n negative movement {}".format(wire0_size_tup[0], wire0_size_tup[1]))
    wire1_size_tup = _max_dim(wire1)
    print("Wire1\n positive movement {}\n negative movement {}".format(wire1_size_tup[0], wire1_size_tup[1]))
    
    # Calculate a starting position that will work for both wires
    start_pos = [0, 0]
    start_pos[0] = abs(min([wire0_size_tup[1][0], wire1_size_tup[1][0]])) + padding
    start_pos[1] = abs(min([wire0_size_tup[1][1], wire1_size_tup[1][1]])) + padding
    print("Starting Position = {}".format(start_pos))
    
    # Calculate the size of the array required given the starting position and the expected movements.
    array_size = [0, 0]
    array_size[0] = max([wire0_size_tup[0][0], wire1_size_tup[0][0]]) + padding
    array_size[1] = max([wire0_size_tup[0][1], wire1_size_tup[0][1]]) + padding
    array_size[0] += start_pos[0]
    array_size[1] += start_pos[1]
    print("Grid Size = {}".format(array_size))

    # Create numpy arrays to map the movement data onto
    wire0_array = np.zeros(array_size, int)
    wire1_array = np.zeros(array_size, int)
    _wire_array(wire0, wire0_array, start_pos)
    _wire_array(wire1, wire1_array, start_pos)
    # Find where the crossings occur by multiplying (0's in unused spaces)
    crossing_array = wire0_array * wire1_array
    # Numpy way to get the non-zero indices.
    crossing_tuple = np.nonzero(crossing_array)
    # Table for readability
    table = asciitable.AsciiTable(getattr(log, "result"))
    table.setColumnName(0, " Crossing Coordinate ")
    table.setColumnName(1, " Manhattan Distance ")
    table.setColumnName(2, " Wire0 Step Count ")
    table.setColumnName(3, " Wire1 Step Count ")
    table.setColumnName(4, " Total Step Count ")
    row = 0
    distances = []
    total_step_counts = []
    # Walk the intersection coordinates collecting the data we need.
    for i, j in zip(crossing_tuple[0], crossing_tuple[1]):
        # Calculate the Manhattan distance
        distances.append(abs(i-start_pos[0]) + abs(j-start_pos[1]))
        # Sum the step count from both wire array maps.
        total_step_counts.append( wire0_array[i,j] + wire1_array[i,j])
        table.writeRow(row, ["{}".format([i,j]), distances[-1], wire0_array[i,j], wire1_array[i,j], total_step_counts[-1]])
        row += 1
    table.colJustify = ['center' for x in table.colJustify]
    table.show()
    print("Minimum Manhattan Distance = {}".format(min(distances)))
    print("Minimum Step Count = {}".format(min(total_step_counts)))


def _day4(start,stop):
    """
    Password search
    """
    counter1 = 0
    counter2 = 0
    num = start - 1
    password_length = 6
    if start < 1 * (10**(password_length-1)) or stop > ((1 * 10 ** password_length) - 1):
        raise Exception("Password must be 6 digits")
    while num < stop:
        num += 1
        flag = False
        # Split the number into an list of digits.
        num_str = str(num)
        # From left to right the digits never decrease
        for i in range(password_length - 1):
            if num_str[i] > num_str[i+1]:
                # Decreasing number found, set all remaining numbers to the same value.
                num_str = num_str[:i] + (num_str[i] * (password_length - i))
                num = int(num_str)
                # Extra range check
                if num > stop:
                    flag = True
        if flag: break
        # Part 1 - Seach for any two adjacent digits which match
        for i in range(password_length - 1):
            if num_str[i] == num_str[i+1]:
                flag = True
                break
        if flag:
            counter1 += 1
        # Part 2 - Check that the matching digits are not part of a larger set of matching digits.
        flag = False
        for i in range(password_length - 1):
            if num_str[i] == num_str[i+1]:
                if i == 0: before = -1
                else: before = num_str[i-1]
                if (i+2) == password_length: after = -1
                else: after = num_str[i+2]
                if num_str[i] not in [before, after]:
                    #print("{}, index:{} before:{}, after:{}".format(num, i, before, after))
                    flag = True
                    break
        if flag: counter2 += 1
    log.result("Part 1 - Found {} valid passwords".format(counter1))
    log.result("Part 2 - Found {} valid passwords".format(counter2))

def _day5():
    opcode_tup = _pull_puzzle_input(5, ',', int)
    comp = Computer()
    comp.load_mem(opcode_tup)
    comp.data_in = [1]
    comp.go()
    # Last output is the diagnostic code.
    print("Part 1\n TEST Output: {}\n Diagnostic Code: {}".format(comp.data_out[:-1],comp.data_out[-1]))
    comp.reset()
    comp.data_in = [5]
    comp.go()
    print("Part 2\n TEST Output: {}\n Diagnostic Code: {}".format(comp.data_out[:-1],comp.data_out[-1]))




def _day6():
    """
    Another tree puzzle, I should remember to write up a tree function before the AoC starts each year.
    """
    orbit_dict = {}
    thing_set = set()
    orbit_map = _pull_puzzle_input(6, '\n', to_tup)
    # Example data
    #orbit_map = [('COM','B'),('B','C'),('C','D'),('D','E'),('E','F'),('B','G'),('G','H'),('D','I'),('E','J'),('J','K'),('K','L')]
    #orbit_map = [('COM','B'),('B','C'),('C','D'),('D','E'),('E','F'),('B','G'),('G','H'),('D','I'),('E','J'),('J','K'),('K','L'),('K','YOU'),('I','SAN')]

    # Create a dictionary key is a object and value is the thing it orbits around.
    # Also make a set of all the objects in this solar system
    for orbited, orbiter in orbit_map:
        thing_set.add(orbiter)
        if orbiter in orbit_dict.keys():
            # Can only orbit one thing according to the rules.
            raise Exception("Duplicate? {}".format([orbited, orbiter]))
        orbit_dict[orbiter] = orbited

    # For every planet in the solar system walk backwards to COM each step is a direct or indirect orbit.
    orbit_count = 0
    for thing in thing_set:
        while thing != "COM":
            old_thing = thing
            thing = orbit_dict[thing]
            orbit_count += 1
            #print("{} -> {}".format(old_thing, thing))
    print("Total orbits: {}".format(orbit_count))

    # Map the route from YOU to COM and SAN to COM and remove the common nodes (planets) between the two.
    you_set = set()
    pos = "YOU"
    while pos != "COM":
        you_set.add(pos)
        pos = orbit_dict[pos]
    san_set = set()
    pos = "SAN"
    while pos != "COM":
        san_set.add(pos)
        pos = orbit_dict[pos]
    #print(you_set)
    #print(san_set)
    symmetric_diff = you_set ^ san_set
    #print(symmetric_diff)
    # Subtract 2 because the YOU orbit and the SAN orbit don't count (could also remove YOU and SAN from the list.
    print("Orbital transfers to Santa: {}".format(len(symmetric_diff)-2))


def _day7(program_override=None):
    """
    """
    if program_override is None:
        opcode_tup = _pull_puzzle_input(7, ',', int)
    else:
        opcode_tup = tuple(program_override)
    # example 2
    # opcode_tup = (3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,23,99,0,0)
    # example 3
    # opcode_tup = (3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0)
    # opcode_tup = (3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5)
    comp = Computer()
    comp.load_mem(opcode_tup)

    phase_permutations = list(itertools.permutations([0,1,2,3,4]))
    max_output = [0,[]]
    for phase_list in phase_permutations:
        signals = [0]
        for phase in phase_list:
            signals.append(phase)
            comp.reset()
            comp.data_in = signals
            comp.go()
            signals = comp.data_out
        #log.result("{} = {}".format(phase_list, amp_transfer))
        if signals[0] > max_output[0]:
            max_output[0] = signals[0]
            max_output[1] = phase_list
    print("Part 1 - max: {} phase: {}".format(max_output[0],max_output[1]))
    del comp


    # Part 2
    phase_permutations = list(itertools.permutations([5,6,7,8,9]))
    
    # Examples:
    # example 1
    #opcode_tup = (3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5)
    #phase_permutations = [[9,8,7,6,5]]
    # example 2
    # opcode_tup = (3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,-5,54,1105,1,12,1,53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10)

    # 5 amplifier stages
    amps = []
    for i in range(5):
        temp_comp = Computer()
        temp_comp.load_mem(opcode_tup)
        amps.append(temp_comp)
    for phase_list in phase_permutations:
        # Load up the phase data
        log.info("Load phase information")
        for i in range(5):
            amps[i].reset()
            amps[i].data_in = [phase_list[i]]
            amps[i].go()
            log.info(" amp:{} ip:{} input:{} output:{} mem:{}".format(i,amps[i]._ip,phase_list[i],amps[i].data_out,amps[i]._mem))
        # Now the amps should all be ready for the amplification loop.
        transfer_value = 0
        amplify_flag = True
        log.info("Begin the amplification loop")
        counter = 0
        while amplify_flag:
            log.info("Loop Count: {}".format(counter))
            counter += 1
            for i in range(5):
                if amps[i].terminate:
                    raise Exception("Attempting a run on amplifier {} that has already terminated!".format(i))
                amps[i].data_in = [transfer_value]
                amps[i].go()
                log.info(" amp:{} ip:{} input:{} output:{} mem:{}".format(i,amps[i]._ip,transfer_value,amps[i].data_out,amps[i]._mem))
                transfer_value = amps[i].data_out.pop()
                #log.result("amp{} {} mem: {}".format(i, transfer_value, amps[4]._mem))
                if i == 4 and amps[4].terminate:
                    log.info("Loop terminated")
                    amplify_flag = False
                    break
        if transfer_value > max_output[0]:
            max_output = [transfer_value, phase_list]
    log.result("Part 2 - Signal: {} phase: {}".format(max_output[0],max_output[1]))
    del amps


def _day8():
    sif = _pull_puzzle_input(8, '\n', _dummy)[0]
    dimension = (25,6)
    # Example
    # sif = '122456789012'
    # dimension = (3,2)
    # Example 2
    #sif = '0222112222120000'
    #dimension = (2,2)
    layers = int(len(sif)/(dimension[0]*dimension[1]))
    log.result("Image has {} layers".format(layers))
    image_array = np.empty([layers, dimension[1], dimension[0]], int)
    log.result("array size {}".format(image_array.size))
    index = 0
    for layer in range(layers):
        for y in range(dimension[1]):
            for x in range(dimension[0]):
                image_array[layer, y, x] = int(sif[index])
                index += 1
    fewest_zero = (150,-1)
    for layer in range(layers):
        zeros = image_array[layer].size - np.count_nonzero(image_array[layer])
        if zeros < fewest_zero[0]:
            fewest_zero = (zeros, layer)
    log.result("Part 1")
    log.result("  Layer {} has the fewest zeros".format(fewest_zero[1]))
    answer = np.count_nonzero(image_array[fewest_zero[1]] == 1) * np.count_nonzero(image_array[fewest_zero[1]] == 2)
    log.result("  1's * 2's on layer {} is {}".format(fewest_zero[1],answer))
    # Part 2
    output = np.empty([dimension[1], dimension[0]], int)
    for layer in range(layers-1,-1,-1):
        output = np.where(image_array[layer] != 2, image_array[layer], output)
        #log.result(image_array[layer])
    # Readability loop, might be some better way to display a numpy array...
    log.result("Part 2 - Image")
    for row in range(dimension[1]):
        output_str = '  '
        for col in range(dimension[0]):
            if output[row][col] == 1:
                output_str += '#'
            else:
                output_str += ' '
        log.result(output_str)


def _day9():
    """
    """
    global puzzle_dict
    try:
        opcode_tup = puzzle_dict[9]
    except:
        opcode_tup = _pull_puzzle_input(9, ',', int)
        puzzle_dict[9] = opcode_tup
    # Example 1
    # opcode_tup = (109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99)
    # Example 2
    #opcode_tup = (1102,34915192,34915192,7,4,7,99,0)
    comp = Computer()
    comp.load_mem(opcode_tup)
    comp.data_in = [1]
    comp.go()
    log.result("Part 1 - BOOST keycode: {}".format(comp.data_out[-1]))
    # log.result(" Cache hit: {}".format(comp._hit))
    comp.reset()
    comp.data_in = [2]
    comp.go()
    log.result("Part 2 - BOOST keycode: {}".format(comp.data_out[-1]))
    # log.result(" Cache hit: {}".format(comp._hit))


def _str_to_list(the_string):
    """
    parsing function for day 10.
    """
    return list(the_string)


def _roid_removal_angles(roid_map, test_pos):
    """
    Re-do the search but use angles this time?
    """
    import cmath
    roid_coords = np.transpose(np.nonzero(roid_map == '#'))
    detected_roids = {}
    occluded_roids = []
    for roid in roid_coords:
        if list(roid) == list(test_pos):
            #log.debug("Center {}".format(roid))
            continue
        roid_complex = complex((roid[0]-test_pos[0]),(roid[1]-test_pos[1]))
        r, phi = cmath.polar(roid_complex)
        #log.debug("{} - r:{} phi:{}".format(roid,r,phi))
        if phi not in detected_roids.keys():
            detected_roids[phi] = [r, roid]
        elif detected_roids[phi][0] < r:
            occluded_roids.append(roid)
        else:
            occluded_roids.append(detected_roids[phi][1])
            detected_roids[phi] = [r, roid]
    #log.debug("detected")
    #for k, v in detected_roids.items():
    #    log.debug("{} phi:{} r:{}".format(v[1], k, v[0]))
    #log.debug("occluded")
    #for roid in occluded_roids:
    #    log.debug(roid)
    roid_map[test_pos] = "B"
    for roid in occluded_roids:
        roid_map[tuple(roid)] = "*"


def _print_roid_map(roid_map):
    """
    Print the roid map, highlight the special roid if given.
    """
    # Table for readability
    roid_table = asciitable.AsciiTable(getattr(log, "result"))
    # No borders
    for key in roid_table.border:
        roid_table.border[key] = ['', 0]
    # Highlight colors
    magenta = slt.CreateLogColor('ired')
    blue = slt.CreateLogColor('iblue')
    white = slt.CreateLogColor('iwhite')
    # The loop!
    for line in range(len(roid_map)):
        line_list = []
        for col in range(len(roid_map[0])):
            if roid_map[line][col] == "*":
                line_list.append(magenta("*"))
            elif roid_map[line][col] == "#":
                line_list.append(blue("#"))
            else:
                line_list.append(roid_map[line][col])
        # line_list.append(" {}".format(np.count_nonzero(roid_map[line] == "#")))
        roid_table.writeRow(line, line_list)
    roid_table.colJustify = ['center' for x in roid_table.colJustify]
    roid_table.show()


def _roid_mapping(roid_map, test_pos):
    """
    Create an orderd
    """
    import cmath
    roid_coords = np.transpose(np.nonzero(roid_map == '#'))
    polar_roid_map = {}
    for roid in roid_coords:
        if list(roid) == list(test_pos):
            #log.debug("Center {}".format(roid))
            continue
        roid_complex = complex((test_pos[0]-roid[0]),(roid[1]-test_pos[1]))
        #roid_complex = complex((test_pos[1]-roid[1]),(test_pos[0]-roid[0]))
        #roid_complex = complex((roid[1]-test_pos[1]),(test_pos[0]-roid[0]))
        r, phi = cmath.polar(roid_complex)
        if phi < 0:
            phi = 2 * cmath.pi + phi
        #phi = phi + (2*cmath.pi)
        #phi = phi * -1
        #log.debug("{} - r:{} phi:{}".format(roid,r,phi))
        if phi not in polar_roid_map.keys():
            polar_roid_map[phi] = {r: roid}
        else:
            polar_roid_map[phi][r] = roid
    return polar_roid_map


def _day10(test=False):
    """
    """
    if test is False:
        roid_map = _pull_puzzle_input(10, '\n', _str_to_list)
        roid_map = npa(roid_map, str)
        roid_coords = np.transpose(np.nonzero(roid_map == '#'))
    else:
        # Example 1
        ex = ['......#.#.','#..#.#....','..#######.','.#.#.###..','.#..#.....','..#....#.#','#..#....#.','.##.#..###','##...#..#.','.#....####']
        # Example 2
        ex = ['#.#...#.#.','.###....#.','.#....#...','##.#.#.#.#','....#.#.#.','.##..###.#','..#...##..','..##....##','......#...','.####.###.']
        # Example 3
        ex = ['.#..#..###','####.###.#','....###.#.','..###.##.#','##.##.#.#.','....###..#','..#.#..#.#','#..#.#.###','.##...##.#','.....#.#..']
        # Example 4
        ex = ['.#..##.###...#######','##.############..##.','.#.######.########.#','.###.#######.####.#.','#####.##.#.##.###.##','..#####..#.#########','####################','#.####....###.#.#.##','##.#################','#####.##.###..####..','..######..##.#######','####.##.####...##..#','.#####..#.######.###','##...#.##########...','#.##########.#######','.####.#.###.###.#.##','....##.##.###..#####','.#.#.###########.###','#.#.#.#####.####.###','###.##.####.##.#..##']
        #ex = ['.'*9]*9
        #ex[0]='....#....'
        #ex[3]='#........'
        #ex[4]='#...#...#'
        #ex[7]='....#....'
        #ex[8]='....#....'
        #ex_s =20
        #ex = ['#'*ex_s] * ex_s
        ex = list(list(x) for x in ex)
        roid_map = npa(ex)
        roid_coords = np.transpose(np.nonzero(roid_map == '#'))
        roid_coords=[[13,11]]
        #roid_coords=[[4,4]]

    log.result("Part 1")
    log.result(" Roid map has {} roids".format(np.count_nonzero(roid_map == "#")))
    _print_roid_map(roid_map)
    max_roids = 0
    max_pos = None
    max_map = copy.copy(roid_map)
    max_map.fill("")
    polar_roid_map = {}
    for roid in roid_coords:
        visible_roids = 0
        polar_roid_map = _roid_mapping(roid_map, roid)
        occluded_roid_map = copy.deepcopy(polar_roid_map)
        for phi in sorted(polar_roid_map.keys()):
            visible_roids += 1
            occluded_roid_map[phi].pop(sorted(occluded_roid_map[phi].keys())[0])
        if visible_roids > max_roids:
            max_roids = visible_roids
            max_pos = roid
            max_roid_map = copy.deepcopy(polar_roid_map)
            max_occluded_map = copy.deepcopy(occluded_roid_map)
    for phi, roids in max_occluded_map.items():
        for r, roid in roids.items():
            roid_map[tuple(roid)] = "*"
    roid_map[tuple(max_pos)] = "X"

    log.result(" Best is at {} with {} roids detectable".format(list(reversed(max_pos)), max_roids))
    _print_roid_map(roid_map)
    log.result("Part 2")
    lazered_roids = []
    # Walk the roid map in order as the lazer sweeps across them appending to a list.
    continue_flag = True
    while len(max_roid_map) != 0 and continue_flag:
        for phi in sorted(max_roid_map.keys()):
            if len(max_roid_map[phi]) == 0:
                max_roid_map.pop(phi)
            else:
                lazered_roids.append(max_roid_map[phi].pop(sorted(max_roid_map[phi].keys())[0]))
            if len(lazered_roids) == 200:
                continue_flag = False
                break
    lazered = list(reversed(lazered_roids[99]))
    log.result(" 100th roids lazered will be {} = {}".format(lazered, (lazered[0] * 100 + lazered[1])))
    lazered = list(reversed(lazered_roids[199]))
    log.result(" 200th roids lazered will be {} = {}".format(lazered, (lazered[0] * 100 + lazered[1])))


def _day11():
    """
    """
    global puzzle_dict
    try:
        opcode_tup = puzzle_dict[11]
    except:
        opcode_tup = _pull_puzzle_input(11, ',', int)
        puzzle_dict[11] = opcode_tup
    opcode_str = '3,8,1005,8,328,1106,0,11,0,0,0,104,1,104,0,3,8,1002,8,-1,10,1001,10,1,10,4,10,1008,8,0,10,4,10,1001,8,0,29,1,104,7,10,3,8,1002,8,-1,10,101,1,10,10,4,10,1008,8,0,10,4,10,1001,8,0,55,1,2,7,10,1006,0,23,3,8,102,-1,8,10,1001,10,1,10,4,10,1008,8,0,10,4,10,1001,8,0,84,1006,0,40,1,1103,14,10,1,1006,16,10,3,8,102,-1,8,10,101,1,10,10,4,10,108,1,8,10,4,10,1002,8,1,116,1006,0,53,1,1104,16,10,3,8,102,-1,8,10,101,1,10,10,4,10,1008,8,1,10,4,10,102,1,8,146,2,1104,9,10,3,8,102,-1,8,10,101,1,10,10,4,10,1008,8,1,10,4,10,1001,8,0,172,1006,0,65,1,1005,8,10,1,1002,16,10,3,8,102,-1,8,10,1001,10,1,10,4,10,108,0,8,10,4,10,102,1,8,204,2,1104,9,10,1006,0,30,3,8,102,-1,8,10,101,1,10,10,4,10,108,0,8,10,4,10,102,1,8,233,2,1109,6,10,1006,0,17,1,2,6,10,3,8,102,-1,8,10,101,1,10,10,4,10,108,1,8,10,4,10,102,1,8,266,1,106,7,10,2,109,2,10,2,9,8,10,3,8,102,-1,8,10,101,1,10,10,4,10,1008,8,1,10,4,10,1001,8,0,301,1,109,9,10,1006,0,14,101,1,9,9,1007,9,1083,10,1005,10,15,99,109,650,104,0,104,1,21102,1,837548789788,1,21101,0,345,0,1106,0,449,21101,0,846801511180,1,21101,0,356,0,1106,0,449,3,10,104,0,104,1,3,10,104,0,104,0,3,10,104,0,104,1,3,10,104,0,104,1,3,10,104,0,104,0,3,10,104,0,104,1,21101,235244981271,0,1,21101,403,0,0,1105,1,449,21102,1,206182744295,1,21101,0,414,0,1105,1,449,3,10,104,0,104,0,3,10,104,0,104,0,21102,837896937832,1,1,21101,0,437,0,1106,0,449,21101,867965862668,0,1,21102,448,1,0,1106,0,449,99,109,2,22102,1,-1,1,21101,40,0,2,21102,1,480,3,21101,0,470,0,1106,0,513,109,-2,2106,0,0,0,1,0,0,1,109,2,3,10,204,-1,1001,475,476,491,4,0,1001,475,1,475,108,4,475,10,1006,10,507,1101,0,0,475,109,-2,2106,0,0,0,109,4,1201,-1,0,512,1207,-3,0,10,1006,10,530,21102,1,0,-3,22102,1,-3,1,21201,-2,0,2,21102,1,1,3,21102,549,1,0,1106,0,554,109,-4,2105,1,0,109,5,1207,-3,1,10,1006,10,577,2207,-4,-2,10,1006,10,577,21202,-4,1,-4,1106,0,645,21202,-4,1,1,21201,-3,-1,2,21202,-2,2,3,21101,596,0,0,1106,0,554,21201,1,0,-4,21102,1,1,-1,2207,-4,-2,10,1006,10,615,21101,0,0,-1,22202,-2,-1,-2,2107,0,-3,10,1006,10,637,22102,1,-1,1,21101,637,0,0,105,1,512,21202,-2,-1,-2,22201,-4,-2,-4,109,-5,2106,0,0'
    opcode_list = opcode_str.split(",")
    opcode_map = []
    for i in opcode_list:
        opcode_map.append(int(i))
    opcode_tup = tuple(opcode_map)
    # Dictionary to translate the movement directions.
    move_dict = {
                    (-1,  0): {0:npa([0,-1]), 1:npa([0,1])},  # up
                    ( 1,  0): {0:npa([0,1]), 1:npa([0,-1])},  # down
                    ( 0,  1): {0:npa([-1,0]), 1:npa([1,0])},  # right
                    ( 0, -1): {0:npa([1,0]), 1:npa([-1,0])}  # left
                }
    move_name = {
                    (-1,  0): 'up',
                    ( 1,  0): 'down',
                    ( 0,  1): 'right',
                    ( 0, -1): 'left'
                }
    bot_pos = npa([0,0])
    bot_move = npa([-1,0])
    hull_dict = {(0,0):1}
    comp = Computer()
    comp.load_mem(opcode_tup)
    counter = 0
    while comp.terminate is not True:
        if tuple(bot_pos) not in hull_dict:
            color = 0
        else:
            color = hull_dict[tuple(bot_pos)]
        comp.data_in = [color]
        comp.go()
        hull_dict[tuple(bot_pos)] = comp.data_out.pop(0)
        #_print_hull(hull_dict)
        move_command = comp.data_out.pop(0)
        next_move = move_dict[tuple(bot_move)][move_command]
        new_pos = bot_pos + next_move
        # log.result("s:{} d:{} c:{} e:{} d:{}".format(bot_pos, bot_move, move_command, new_pos, next_move))
        bot_move = next_move
        bot_pos = new_pos
    log.result("Panels painted once: {}".format(len(hull_dict.keys())))

    _print_hull(hull_dict)
    log.result("comp mem size = {}".format(comp._mem_size))

def _print_hull(hull_dict):
        # Size
    max_p_v = 0
    max_n_v = 0
    max_p_h = 0
    max_n_h = 0
    for pos in hull_dict.keys():
        if pos[0] > max_p_v: max_p_v = pos[0]
        if pos[0] < max_n_v: max_n_v = pos[0]
        if pos[1] > max_p_h: max_p_h = pos[1]
        if pos[1] < max_n_h: max_n_h = pos[1]
    log.result("Required array size: [{}, {}] to [{},{}]".format(max_p_v,max_p_h,max_n_v,max_n_h))
    art = np.empty([(max_p_v-max_n_v+1),(max_p_h-max_n_h+1)],str)
    for pos, color in hull_dict.items():
        if color == 1: art[pos] = "#"
        else: art[pos] = " "

    art_table = asciitable.AsciiTable(getattr(log, "result"))
    # No borders
    for key in art_table.border:
        art_table.border[key] = ['', 0]
    # The loop!
    for row in range(max_p_v-max_n_v+1):
        art_table.writeRow(row, list(art[row]))
    # art_table.colJustify = ['center' for x in art_table.colJustify]
    art_table.show()


def _day12():
    """
    Part 1 solution was too slow.
    """
    moon_list = get_input(12, '\n', _dummy)
    # Example 1
    # moon_list = ['<x=-1, y=0, z=2>','<x=2, y=-10, z=-7>','<x=4, y=-8, z=8>','<x=3, y=5, z=-1>']
    # Example 2
    # moon_list = ['<x=-8, y=-10, z=0>','<x=5, y=5, z=10>','<x=2, y=-7, z=3>','<x=9, y=-8, z=-3>']
    number_of_moons = len(moon_list)
    system_array = np.zeros([number_of_moons,6],int)
    for i in range(number_of_moons):
        moon_position_list = re.split('[=,]', moon_list[i].rstrip('>'))
        for j in range(3):
            system_array[i][j] = int(moon_position_list[(j * 2) + 1])
    log.result("Initial System")
    log.result(system_array)
    axis_periods = []
    # Find the reset for one axis
    time = 0
    temp_array = copy.copy(system_array)
    axis_list = [0, 1, 2]
    axis_dict = {0:set(), 1:set(), 2:set()}
    search_flag = True
    while search_flag:
        # Find a moon period
        for axis in axis_list:
            axis_bytes = npa([temp_array[:,axis],temp_array[:,axis + 3]]).tobytes()
            if axis_bytes in axis_dict[axis]:
                axis_periods.append(time)
                axis_list.remove(axis)
                if len(axis_list) == 0:
                    search_flag = False
                    break
            axis_dict[axis].add(axis_bytes)
        # Gravity
        for moon in range(number_of_moons):
            temp_array[moon,3:] += (temp_array[moon,:3] < temp_array[:,:3]).sum(axis=0)
            temp_array[moon,3:] -= (temp_array[moon,:3] > temp_array[:,:3]).sum(axis=0)
        temp_array[:,:3] += temp_array[:,3:]
        time += 1
    #log.result("System returns to original state after {} time units".format(np.gcd.reduce(moon_periods)*max(moon_periods)))
    log.result("Axis periods: {}".format(axis_periods))
    log.result("Position & velocity repeat at {}".format(np.lcm.reduce(axis_periods,dtype='int64')))
    return


def _day12_part1():
    """
    Day 12 Moons!
    """
    moon_list = get_input(12, '\n', _dummy)
    # Example 1
    #moon_list = ['<x=-1, y=0, z=2>','<x=2, y=-10, z=-7>','<x=4, y=-8, z=8>','<x=3, y=5, z=-1>']
    # Example 2
    #moon_list = ['<x=-8, y=-10, z=0>','<x=5, y=5, z=10>','<x=2, y=-7, z=3>','<x=9, y=-8, z=-3>']
    moon_dict = {}
    for moon in moon_list:
        moon_position_list = re.split('[=,]', moon.rstrip('>'))
        moon_dict[moon] = {
                'position':npa([int(moon_position_list[1]),int(moon_position_list[3]),int(moon_position_list[5])]),
                'velocity':npa([0,0,0]),
                'new_position':npa([0,0,0])
                }
    universe_set = set()
    position_str = ''
    velocity_str = ''
    for moon in moon_list:
        position_str += "{}".format(moon_dict[moon]['position'])
        velocity_str += "{}".format(moon_dict[moon]['velocity'])
    universe_set.add((position_str, velocity_str))
    log.result("Initial Positions:")
    # Print
    for moon, moon_data in moon_dict.items():
        log.result(" pos=[{:2d},{:2d},{:2d}] vel=[{:2d},{:2d},{:2d}]".format(moon_data['position'][0],moon_data['position'][1],moon_data['position'][2], moon_data['velocity'][0],moon_data['velocity'][1],moon_data['velocity'][2]))
    # for time in range(1,11):
    #for time in range(1,101):
    energy_print = 1000
    time = 0
    while True:
        time += 1
        # Apply gravity
        for a,b in itertools.permutations(moon_dict, 2):
            for dimension in range(3):
                if moon_dict[b]['position'][dimension] > moon_dict[a]['position'][dimension]:
                    moon_dict[a]['velocity'][dimension] += 1
                elif moon_dict[b]['position'][dimension] < moon_dict[a]['position'][dimension]:
                    moon_dict[a]['velocity'][dimension] -= 1
            moon_dict[a]['new_position'] = moon_dict[a]['position'] + moon_dict[a]['velocity']
        # Gravity application complete, update positions.
        for moon, moon_data in moon_dict.items():
            moon_data['position'] = moon_data['new_position']
        match = True
        position_str = ''
        velocity_str = ''
        if time == energy_print:
            log.result("Final at {}:".format(time))
            # Final results
            energy = 0
            for moon, moon_data in moon_dict.items():
                moon_energy = (np.sum(np.absolute(moon_data['position'])) * np.sum(np.absolute(moon_data['velocity'])))
                log.result(" pos=[{:2d},{:2d},{:2d}] vel=[{:2d},{:2d},{:2d}] - {}".format(moon_data['position'][0],moon_data['position'][1],moon_data['position'][2], moon_data['velocity'][0],moon_data['velocity'][1],moon_data['velocity'][2],moon_energy))
                energy += moon_energy
            log.result(" total energy is {}".format(energy))
            break

def _day13_draw(stdscr):
    """
    """
    shape_dict = {0:' ', 1:'\u2588', 2:'#', 3:'\u2501', 4:'\u25CF'}
    stdscr.keypad(True)
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)
    color_dict = {
            2: curses.color_pair(1),
            3: curses.color_pair(1),
            4: curses.color_pair(2),
            5: curses.color_pair(2),
            6: curses.color_pair(2),
            7: curses.color_pair(3),
            8: curses.color_pair(3),
            9: curses.color_pair(3),
            10: curses.color_pair(4),
            11: curses.color_pair(4),
            12: curses.color_pair(4),
            13: curses.color_pair(5),
            14: curses.color_pair(5),
            15: curses.color_pair(5),
            16: curses.color_pair(6),
            17: curses.color_pair(6),
            18: curses.color_pair(6)
            }
    opcode_tup = get_input(13, ',', int)
    comp = Computer()
    comp.load_mem(opcode_tup)
    comp.write_mem(0,2)
    #comp.data_in = [0]
    comp.go()
    maxy = 0
    stdscr.clear()
    print_blocks = True
    while True:
        blocks = 0
        while comp.data_out:
            x, y, tile_id = comp.data_out[:3]
            if y > maxy: maxy = y
            del comp.data_out[:3]
            if x == -1 and y == 0:
                stdscr.addstr(maxy + 2, 0, "Score: {}".format(tile_id))
            elif tile_id == 2:
                blocks += 1
                stdscr.addstr(y, x, shape_dict[tile_id], color_dict.get(y, None))
            else:
                if tile_id == 3:
                    paddle = x
                if tile_id == 4:
                    ball = x
                stdscr.addstr(y, x, shape_dict[tile_id])
        if print_blocks:
            stdscr.addstr(maxy + 1, 0, "Inital blocks: {}".format(blocks))
            print_blocks = False
        stdscr.refresh()
        if paddle < ball:
            comp.data_in.append(1)
        elif paddle > ball:
            comp.data_in.append(-1)
        else:
            comp.data_in.append(0)
        comp.go()
        if len(comp.data_out) == 0:
            stdscr.getkey()
            return
        curses.napms(1)


def _day13():
    """
    """
    curses.wrapper(_day13_draw)


def day14(example=False):
    """
    """
    reaction_list = get_input(14, '\n', _dummy)
    if example:
        reaction_list  = "10 ORE => 10 A\n1 ORE => 1 B\n7 A, 1 B => 1 C\n7 A, 1 C => 1 D\n7 A, 1 D => 1 E\n7 A, 1 E => 1 FUEL".split('\n')





def _test_direction(comp, direction, maze_dict):
    """
    Test a direction and back up if empty.
    """
    movement_dict = {
        ( 0, -1): 1,  # North
        ( 0,  1): 2,  # South
        (-1,  0): 3,  # West
        ( 1,  0): 4   # East
        }
    reverse_dict = {
        ( 0, -1): ( 0,  1),
        ( 0,  1): ( 0, -1),
        (-1,  0): ( 1,  0),
        ( 1,  0): (-1,  0)
        }
    comp.data_in.append(movement_dict[direction])
    comp.go()

    if len(comp.data_out) != 0:
        output = comp.data_out.pop()
        maze_dict['test'] = tuple((npa(maze_dict['robot']) + npa(direction)))
        if output == 0:  # Wall
            maze_dict['wall'].add(maze_dict['test'])
        elif output in [1,2]:  # Moved
            maze_dict['empty'].add(maze_dict['test'])
            if output == 2 and maze_dict['o2'] is None:
                maze_dict['o2'] = copy.copy(maze_dict['test'])
            # Move back
            comp.data_in.append(movement_dict[reverse_dict[direction]])
            comp.go()
        _screen_adj(maze_dict)


def _screen_adj(maze_dict):
    # Shift when we hit the edge of the screen
    left_correction = npa([1,0])
    up_correction = npa([0,1])
    correction = None
    if maze_dict['test'][0] <= 0:
        correction = left_correction
    if maze_dict['test'][1] <= 0:
        correction = up_correction
    if correction is not None:
        # Off screen to the left
        temp_set = set()
        for wall in maze_dict['wall']:
            temp_set.add(tuple((npa(wall) + correction)))
        maze_dict['wall'] = copy.copy(temp_set)
        temp_set = set()
        for empty in maze_dict['empty']:
            temp_set.add(tuple((npa(empty) + correction)))
        maze_dict['empty'] = copy.copy(temp_set)
        del temp_set
        maze_dict['robot'] = tuple((npa(maze_dict['robot']) + correction))
        maze_dict['start'] = tuple((npa(maze_dict['start']) + correction))
        if maze_dict['o2'] is not None:
            maze_dict['o2'] = tuple((npa(maze_dict['o2']) + correction))
    return


def _o2_fill(scr, maze_dict):
    empty_set = copy.copy(maze_dict['empty'])
    filled_set = set([maze_dict['o2']])
    fill_time = 0
    adjacent = [npa([0,1]), npa([0,-1]), npa([1,0]), npa([-1,0])]
    scr.addstr(0, 0, "                 ")
    scr.refresh()
    while len(empty_set):
        temp_set = set()
        for filled_cell in filled_set:
            for adj in adjacent:
                check = tuple(npa(filled_cell) + adj)
                if check in empty_set:
                    temp_set.add(check)
                    empty_set.remove(check)
        filled_set = filled_set.union(temp_set)
        fill_time += 1
        scr.addstr(0, 0, "Fill Time:{}".format(fill_time))
        for filled_cell in filled_set:
            scr.addstr(*reversed(filled_cell), "@", curses.color_pair(3))
        scr.refresh()
        curses.napms(50)
    scr.getch()


def _run15(scr):
    """
    Day 15 robot.
    """
    scr.clear()
    scr.refresh()
    maxsize = scr.getmaxyx()
    scr.resize(maxsize[0],maxsize[1])
    shape_dict = {
            1:'\u2191',  # North
            2:'\u2193',  # South
            3:'\u2190',  # West
            4:'\u2192'   # East
            }
    movement_dict = {
            ( 0, -1): 1,  # North
            ( 0,  1): 2,  # South
            (-1,  0): 3,  # West
            ( 1,  0): 4   # East
            }


    scr.keypad(True)
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
    opcode_tup = get_input(15, ',', int)
    comp = Computer()
    comp.load_mem(opcode_tup)

    # Maze data dictionary
    maze_dict = {
        'robot': (1,1),
        'dir': (0, -1),
        'start': (1,1),
        'o2': None,
        'test': None,
        'wall': set(),
        'empty': set([tuple([1,1])]),
        'steps': 0,
    }

    # Initalize the map
    scr.clear()
    scr.addstr(*reversed(maze_dict['robot']), shape_dict[movement_dict[maze_dict['dir']]], curses.color_pair(2))
    scr.refresh()
    while True:
        # User input
        key = scr.getch()
        if key == curses.KEY_F2:
            return
        elif key == curses.KEY_F8:
            if maze_dict['o2'] is not None:
                _o2_fill(scr, maze_dict)
        elif key == curses.KEY_F5:
            maze_dict['steps'] = 0
        elif key == curses.KEY_UP:
            maze_dict['dir'] = (0, -1)
        elif key == curses.KEY_DOWN:
            maze_dict['dir'] = (0, 1)
        elif key == curses.KEY_LEFT:
            maze_dict['dir'] = (-1, 0)
        elif key == curses.KEY_RIGHT:
            maze_dict['dir'] = (1, 0)
        else:
            continue
        comp.data_in.append(movement_dict[maze_dict['dir']])
        comp.go()

        if len(comp.data_out) != 0:
            output = comp.data_out.pop()
            maze_dict['test'] = tuple((npa(maze_dict['robot']) + npa(maze_dict['dir'])))
            if output == 0:  # Wall
                maze_dict['wall'].add(maze_dict['test'])
            elif output in [1,2]:  # Moved
                maze_dict['robot'] = maze_dict['test']
                maze_dict['steps'] += 1
                maze_dict['empty'].add(maze_dict['test'])
                if output == 2 and maze_dict['o2'] is None:
                    maze_dict['o2'] = copy.copy(maze_dict['test'])
                # Make mapping easier test all directions.
                for direction in list(movement_dict.keys()):
                    _test_direction(comp, direction, maze_dict)

            # Move the map when we hit the edge.
            _screen_adj(maze_dict)

            # Draw the board
            scr.clear()
            scr.addstr(0, 0, "Steps:{}".format(maze_dict['steps']))
            for wall in maze_dict['wall']:
                scr.addstr(*reversed(wall), '\u2588')
            for empty in maze_dict['empty']:
                scr.addstr(*reversed(empty), '.')
            scr.addstr(*reversed(maze_dict['start']), "S", curses.color_pair(1))
            scr.addstr(*reversed(maze_dict['robot']), shape_dict[movement_dict[maze_dict['dir']]], curses.color_pair(2))
            if maze_dict['o2'] is not None:
                scr.addstr(*reversed(maze_dict['o2']), "O", curses.color_pair(4))
            #scr.addstr(10, 0, "[{},{}]".format(robot_pos[0],robot_pos[1]))
            #scr.addstr(12,0, "{}".format(wall_list))
            #scr.addstr(13,0, "{}".format(empty_list))
            scr.refresh()


def _day15():
    """
    """
    curses.wrapper(_run15)

def _day16_part2():
    from itertools import cycle, accumulate

    signal = get_input(16, '\n', _dummy)
    signal = signal[0]
    # Example
    # signal = "03036732577212944063491565474664"
    signal_list = [int(i) for i in signal]
    offset = int(signal[:7])
    length = 10000 * len(signal_list) - offset
    i = cycle(reversed(signal_list))
    a = [next(i) for _ in range(length)]
    for _ in range(100):
        a = [n % 10 for n in accumulate(a)]
    return "".join(map(str, a[-1:-9:-1]))


def _day16_part1():
    signal = get_input(16, '\n', _dummy)
    # Examples
    # signal = ["12345678"]
    # signal = ["80871224585914546619083218645595"]
    # signal = ["19617804207202209144916044189917"]
    signal_array = npa(list(signal[0]), int)
    base_pattern = npa([0, 1, 0, -1],int)
    next_pattern = np.zeros(signal_array.size, int)
    for phase in range(100):
        for i in range(signal_array.size):
            j = 0
            k = 0
            test = np.empty(signal_array.size * 2, int)
            while j < signal_array.size + 1:
                end = j + i + 1
                test[j:end] = base_pattern[k]
                j = end
                k = (k + 1) % 4
            next_pattern[i] = abs(sum(signal_array * test[1:signal_array.size + 1])) % 10
        signal_array = np.copy(next_pattern)
    print("Part1: phase:{} - {}".format(phase+1, "".join(map(str,signal_array[:8]))))


def _day17_scr(scr, data_list, intersections=set()):
    """
    Curses drawing for day 17.
    """

    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    scr.clear()
    scr.refresh()
    maxsize = scr.getmaxyx()
    scr.resize(maxsize[0],maxsize[1])
    scr.keypad(True)
    curses.curs_set(0)
    scaffold = set()
    
    # Initialize the map
    x = 0
    y = 0
    for character in data_list:
        if character == 10:
            y += 1
            x = 0
        else:
            position = tuple([x,y])
            scr.addstr(y, x, chr(character))
            if character == 94:
                robot = npa((x,y))
            if character == 35:
                if position in intersections:
                    scr.addstr(y, x, 'O', curses.color_pair(1))
                scaffold.add(position)
            x += 1
    scr.refresh()
    if len(intersections) != 0:
        scr.getch()
    return scaffold, robot


def _day17_direction(position, direction, scaffold):
    """
    Given a position and a direction decide the next movement.
    """
    directions = {
        "U" : tuple([position[0],      position[1] - 1]),
        "D" : tuple([position[0],      position[1] + 1]),
        "L" : tuple([position[0] - 1, position[1]]),
        "R" : tuple([position[0] + 1, position[1]])
    }
    rev_direction = {
        "U": "D",
        "D": "U",
        "R": "L",
        "L": "R"
    }
    for new_direction, new_position in directions.items():
        #log.result("dir {} {}".format(direction, new_position))
        if new_position in scaffold and new_direction != rev_direction[direction]:
            return new_direction, list(new_position)
    else:
        return "S", position

def _day17_intersection(position, scaffold):
    """
    Given a position check to see if it is an intersection.
    """
    up =    tuple([position[0],      position[1] + 1])
    down =  tuple([position[0],      position[1] - 1])
    left =  tuple([position[0] - 1, position[1]])
    right = tuple([position[0] + 1, position[1]])
    for direction in [up, down, left, right]:
        if direction not in scaffold:
            return False
    return True

def day17():

    turn_dict = {
        "R": {"U": "L",
              "D": "R"},
        "L": {"U": "R",
              "D": "L"},
        "U": {"R": "R",
              "L": "L"},
        "D": {"R": "L",
              "L": "R"},
        }    
    robot_dir = (0,-1)
    opcode_tup = get_input(17, ',', int)
    comp = Computer()
    comp.load_mem(opcode_tup)
    comp.go()
    scaffold, robot = curses.wrapper(_day17_scr, comp.data_out)
    intersections = set()
    answer = 0
    for position in scaffold:
        if _day17_intersection(position, scaffold):
            intersections.add(position)
            answer += (position[0] * position[1])
    curses.wrapper(_day17_scr, comp.data_out, intersections)
    log.result("Part1: Sum of the alignment parameters: {}".format(answer))
    log.result("Robot: {}".format(robot))
    direction = "U"
    prev_turn = ""
    steps = 0
    debug_count = 0
    while 1:
        if _day17_intersection(robot, scaffold):  # Special case the intersections
            steps += 1
            if direction == "U":
                robot[1] -= 1
            elif direction == "D":
                robot[1] += 1
            elif direction == "R":
                robot[0] += 1
            elif direction == "L":
                robot[0] -= 1
            else:
                log.error("Stop?")
            #log.result("intersection {} {}".format(direction, robot))
            #log.result("move complete")
        else:
            next_direction, robot = _day17_direction(robot, direction, scaffold)
            if next_direction == "S":
                log.result("{}{}".format(prev_turn, steps))
                break
            #print(next_direction, robot)
            if next_direction != direction:
                log.result("{}{}".format(prev_turn, steps))
                prev_turn = turn_dict[direction][next_direction]
                direction = next_direction
                steps = 1
            else:
                steps += 1
        debug_count += 1
    comp.reset()
    comp.write_mem(0,2)
    comp.data_in = list(reversed([65,44,66,44,66,44,65,44,67,44,65,44,65,44,67,44,66,44,67,10,82,44,56,44,76,44,49,50,44,82,44,56,10,82,44,49,50,44,76,44,56,44,82,44,49,48,10,82,44,56,44,76,44,56,44,76,44,56,44,82,44,56,44,82,44,49,48,10,110,10]))
    comp.go()
    log.result("data output: {}".format(comp.data_out[-1]))





def _day19_draw(scr, tractor_dict, scr_size=(60,60), offset=(0,0)):
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    scr.clear()
    scr.refresh()
    curses.resize_term(*scr_size)
    maxsize = scr.getmaxyx()
    #scr.resize(maxsize[0],maxsize[1])
    #scr.addstr(0,0, "{}".format(maxsize))
    #scr.refresh()
    #scr.getch()
    curses.curs_set(0)
    for position, value in tractor_dict.items():
        y = position[0] - offset[0]
        x = position[1] - offset[1]
        if value:
            try:
                scr.addstr(y, x, '\u2588')
            except:
                scr.addstr(0,0, "Trying y,x {} from {}-{}".format([y,x],position,offset))
                scr.getch()
                return
        else:
            try:
                scr.addstr(y, x, '\u00B7')
            except:
                scr.addstr(0,0, "Trying y,x {} from {}-{}".format([y,x],position,offset))
                scr.getch()
                return
        #curses.napms(1)
        scr.refresh()
    scr.refresh()
    scr.getch()


def _day19():
    opcode_tup = get_input(19, ',', int)
    comp = Computer()
    comp.load_mem(opcode_tup)
    tractor_dict = {}
    affected = 0
    for x in range(50):
        for y in range(50):
            position = tuple([y,x])
            comp.reset()
            comp.data_in = [y,x]
            comp.go()
            value = comp.data_out.pop()
            tractor_dict[position] = value
            affected += value
    curses.wrapper(_day19_draw, tractor_dict)
    log.result("Part1: {} affected point".format(affected))
    #position = npa([50,38],int)
    position = npa([44,50],int)
    down = npa([0,1],int)
    right = npa([1,0],int)
    possible = [0,0]
    # Follow the right side of the beam.
    while True:
        # log.result(position)
        comp.reset()
        comp.data_in = list(reversed(position.tolist()))
        comp.go()
        value = comp.data_out.pop()
        if value == 0:
            raise Exception("Lost at {}".format(position))
        # Test the bottom left corner
        comp.reset()
        comp.data_in = list(reversed([position[0]-99,position[1]+99]))
        comp.go()
        value = comp.data_out.pop()
        if value == 1:
            possible = tuple([position[0]-99, position[1]])
            break
        # Move down
        position += down
        value = 1
        while value == 1:
            # Move right until out of the beam.
            position += right
            comp.reset()
            comp.data_in = list(reversed(position.tolist()))
            comp.go()
            value = comp.data_out.pop()
        # Back up one
        position -= right
    log.result("Part2: from {} possible position {} = {}".format(position, possible, possible[0]*10000 + possible[1]))   


def _day21_scr(scr, data_list):
    """
    Curses drawing for day 17.
    """

    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    scr.clear()
    scr.refresh()
    maxsize = scr.getmaxyx()
    scr.resize(maxsize[0],maxsize[1])
    scr.keypad(True)
    curses.curs_set(0)
    
    # Initialize the map
    x = 0
    y = 0
    for character in data_list:
        if character == 10:
            y += 1
            x = 0
        else:
            position = tuple([x,y])
            scr.addstr(y, x, chr(character))
            x += 1
    scr.refresh()
    scr.getch()
    return

def day21():
    opcode_tup = get_input(21, ',', int)
    comp = Computer()
    comp.load_mem(opcode_tup)
    springscript = "NOT A J\nNOT C T\nOR T J\nAND D J\nWALK\n"
    comp.data_in = [ord(ele) for ele in reversed(springscript)]
    comp.go()
    log.result("Damage: {}".format(comp.data_out[-1]))
    curses.wrapper(_day21_scr, comp.data_out)


def day22():
    instructions = get_input(22, '\n', _dummy)
    deck = tuple(list(range(10007)))
    # Example s
    #instructions = "deal with increment 7\ndeal into new stack\ndeal into new stack".split("\n")
    #instructions = ["cut 6","deal with increment 7","deal into new stack"]
    #instructions = ["deal with increment 7","deal with increment 9","cut -2"]
    #instructions = ["deal into new stack","cut -2","deal with increment 7","cut 8","cut -4","deal with increment 7","cut 3",
    #                "deal with increment 9","deal with increment 3","cut -1"]
    #instructions = ["cut 3"]
    #deck = tuple(list(range(10)))
    for instruction in instructions:
        #log.result(instruction)
        if "deal into" in instruction:
            log.result("deal into new stack")
            deck = tuple(reversed(deck))
        elif "deal with" in instruction:
            temp = [-1]*len(deck)
            position = 0
            movement = int(instruction.split(" ")[-1])
            log.result("deal with increment {}".format(movement))
            for card in deck:
                temp[position]=card
                position = (position + movement) % len(temp)
            deck = tuple(temp)
            if deck.count(-1) != 0:
                log.error(deck)
                raise Exception("Problem dealing")
        elif "cut" in instruction:
            cut_value = int(instruction.split(" ")[-1])
            log.result("cut {}".format(cut_value))
            if cut_value < 0:
                cut_value = len(deck) + cut_value
            deck = tuple(deck[cut_value:]+deck[:cut_value])
        else:
            raise Exception("Unknown instruction: {}".format(instruction))
    if len(deck) == 10:
        log.result(deck)
        log.result("Card 9 is at position {}".format(deck.index(9)))
    try:
        log.result("Part 1 card 2019 is at position {}".format(deck.index(2019)))
    except Exception:
        foo=1

    deck = tuple(list(range(119315717514047)))
    for i in range(101741582076661):
        for instruction in instructions:
            #log.result(instruction)
            if "deal into" in instruction:
                log.result("deal into new stack")
                deck = tuple(reversed(deck))
            elif "deal with" in instruction:
                temp = [-1]*len(deck)
                position = 0
                movement = int(instruction.split(" ")[-1])
                log.result("deal with increment {}".format(movement))
                for card in deck:
                    temp[position]=card
                    position = (position + movement) % len(temp)
                deck = tuple(temp)
                if deck.count(-1) != 0:
                    log.error(deck)
                    raise Exception("Problem dealing")
            elif "cut" in instruction:
                cut_value = int(instruction.split(" ")[-1])
                log.result("cut {}".format(cut_value))
                if cut_value < 0:
                    cut_value = len(deck) + cut_value
                deck = tuple(deck[cut_value:]+deck[:cut_value])
            else:
                raise Exception("Unknown instruction: {}".format(instruction))
    log.result("Part 1 card 2019 is at position {}".format(deck.index(2019)))

"""
def cp():
    cProfile.run('aoc.prof()', 'stats')
    p = pstats.Stats('stats')
    p.sort_stats('cumulative').print_stats(20)

"""


def prof():
    for i in range(10):
        _day9()



go = {
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
}
