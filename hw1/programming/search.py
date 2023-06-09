# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


import collections
import heapq
import math
import numpy

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    startpos = maze.getStart()
    queue = collections.deque()
    queue.append((startpos, [startpos]))
    visited = set()
    visited.add(startpos)

    while len(queue) > 0:
        currpos, path = queue.popleft()
        if maze.isObjective(currpos[0], currpos[1]):
            return path

        for nextpos in maze.getNeighbors(currpos[0], currpos[1]):
            if nextpos not in visited:
                nextpath = path.copy()
                nextpath.append(nextpos)
                queue.append((nextpos, nextpath))
                visited.add(nextpos)
    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return astar_multi(maze)


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return astar_multi(maze)


def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    def heuristic(state):
        currpos, unvisited_dots = state
        if len(unvisited_dots) == 0:
            return 0

        min_dist, closest_dot = math.inf, None
        for dot in unvisited_dots:
            if manhattan(currpos, dot) < min_dist:
                min_dist = manhattan(currpos, dot)
                closest_dot = dot

        if len(unvisited_dots) == 1:
            return min_dist

        max_dist = 0
        for dot in unvisited_dots:
            max_dist = max(max_dist, manhattan(closest_dot, dot))
        return min_dist + max_dist

    startpos = maze.getStart()
    start_state = (startpos, set(maze.getObjectives()))
    if maze.isObjective(startpos[0], startpos[1]):
        start_state[1].discard(startpos)

    fringe = []
    heapq.heappush(fringe, (
            heuristic(start_state), start_state, [startpos]
    ))
    closed = set()

    while True:
        if len(fringe) == 0:
            return []

        fvalue, state, path = heapq.heappop(fringe)
        currpos, unvisited_dots = state

        if len(unvisited_dots) == 0:
            return path

        if str(state) not in closed:
            closed.add(str(state))
            for nextpos in maze.getNeighbors(currpos[0], currpos[1]):
                next_unvisited_dots = unvisited_dots.copy()
                next_state = (nextpos, next_unvisited_dots)
                if maze.isObjective(nextpos[0], nextpos[1]):
                    next_state[1].discard(nextpos)

                nextpath = path.copy()
                nextpath.append(nextpos)

                heapq.heappush(fringe, (
                        len(nextpath) - 1 + heuristic(next_state), next_state, nextpath
                ))


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    def heuristic(state):
        currpos, unvisited_dots = state
        return sum(manhattan(currpos, dot) for dot in unvisited_dots)

    startpos = maze.getStart()
    start_state = (startpos, set(maze.getObjectives()))
    if maze.isObjective(startpos[0], startpos[1]):
        start_state[1].discard(startpos)

    fringe = []
    heapq.heappush(fringe, (
            heuristic(start_state), start_state, [startpos]
    ))
    closed = set()

    while True:
        if len(fringe) == 0:
            return []

        fvalue, state, path = heapq.heappop(fringe)
        currpos, unvisited_dots = state

        if len(unvisited_dots) == 0:
            return path

        if str(state) not in closed:
            closed.add(str(state))
            for nextpos in maze.getNeighbors(currpos[0], currpos[1]):
                next_unvisited_dots = unvisited_dots.copy()
                next_state = (nextpos, next_unvisited_dots)
                if maze.isObjective(nextpos[0], nextpos[1]):
                    next_state[1].discard(nextpos)

                nextpath = path.copy()
                nextpath.append(nextpos)

                heapq.heappush(fringe, (
                        len(nextpath) - 1 + heuristic(next_state), next_state, nextpath
                ))

