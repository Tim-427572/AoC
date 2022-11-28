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
    """


def go(day=25):
    switch = {
        1:  _day1,
        # 2:  _day2,
        # 3:  _day3,
        # 4:  _day4,
        # 5:  _day5,
        # 6:  _day6,
        # 7:  _day7,
        # 8:  _day8,
        # 9:  _day9,
        # 10: _day10,
        # 11: _day11,
        # 12: _day12,
        # 13: _day13,
        # 14: _day14,
        # 15: _day15,
        # 16: _day16,
        # 17: _day17,
        # 18: _day18,
        # 19: _day19,
        # 20: _day20,
        # 21: _day21,
        # 22: _day22,
        # 23: _day23,
        # 25: _day25,
    }
    return switch.get(day, "Invalid day")()


import concurrent.futures
import time


#if __name__ == "__main__":
#    loop = asyncio.get_event_loop()
#    loop.run_until_complete(c_thread(loop))
