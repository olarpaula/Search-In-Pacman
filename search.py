# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import random
import math

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def randomSearch ( problem ) :
    current = problem.getStartState () 
    solution = [] 

    while (not (problem.isGoalState(current))) :
      succ = problem.getSuccessors(current)
      no_of_successors = len(succ)
      random_succ_index = int(random.random() * no_of_successors)
      next = succ[random_succ_index]
      current = next[0]
      solution.append(next[1])
    print "The solution is ", solution
    return solution

def depthFirstSearch(problem):
    "*** YOUR CODE HERE ***"

    visited = []
    start_node = problem.getStartState()
    if problem.isGoalState(start_node):
        return []
	
    my_stack = util.Stack()
    my_stack.push((start_node, []))

    while not my_stack.isEmpty():
		curr_node, actions = my_stack.pop()

		if curr_node not in visited:
			visited.append(curr_node)

			if problem.isGoalState(curr_node):
				return actions

			for succesor, next_action, cost in problem.getSuccessors(curr_node):
				new_action = actions + [next_action]
				my_stack.push((succesor, new_action))

    util.raiseNotDefined()

class newDS:
    def __init__(self, name, cost):
        self.name = name 
        self.cost = cost

    def getName(self):
        return self.name

	def getCost(self):
		return self.cost

def breadthFirstSearch(problem):
    "*** YOUR CODE HERE ***"

    visited = []
    start_node = problem.getStartState()
    if problem.isGoalState(start_node):
        return []
	
    my_queue = util.Queue()
    my_queue.push((start_node, []))

    while not my_queue.isEmpty():
		curr_node, actions = my_queue.pop()
		if curr_node not in visited:
			visited.append(curr_node)

			if problem.isGoalState(curr_node):
				return actions

			for succesor, next_action, cost in problem.getSuccessors(curr_node):
				new_action = actions + [next_action]
				my_queue.push((succesor, new_action))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    "*** YOUR CODE HERE ***"
    visited = []
    start_node = problem.getStartState()
    if problem.isGoalState(start_node):
        return []

    my_priorityQ = util.PriorityQueue()
    my_priorityQ.push((start_node, [], 0), 0)

    while not my_priorityQ.isEmpty():
        curr_node, actions, cost = my_priorityQ.pop()

        if curr_node not in visited:
            visited.append(curr_node)

            if problem.isGoalState(curr_node):
                return actions

            for successor, next_action, next_cost in problem.getSuccessors(curr_node):
                new_action = actions + [next_action]
                priority = cost + next_cost
                my_priorityQ.push((successor, new_action, priority), priority)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "*** YOUR CODE HERE ***"
    visited = []
    start_node = problem.getStartState()

    if problem.isGoalState(start_node):
        return []

	"nod, actiune, cost curent nod, prioritate"
    my_priorityQ = util.PriorityQueue()
    my_priorityQ.push((start_node, [], 0), 0)

    while not my_priorityQ.isEmpty():
        curr_node, actions, p_cost = my_priorityQ.pop()

        if curr_node not in visited:
            visited.append(curr_node)

            if problem.isGoalState(curr_node):
                return actions

            for next, action, cost in problem.getSuccessors(curr_node):
                new_action = actions + [action]
                new_cost = p_cost + cost
                new_heuristic = new_cost + heuristic(next, problem)
                my_priorityQ.push((next, new_action, new_cost), new_heuristic)
            
    util.raiseNotDefined()


def depthLimitedSearch(problem, depth):
    """The initial start node is pushed into a Stack, and a reached_depth is being tracked in order to
    limit the depth at which Depth First Search can be applied"""
    start_node = problem.getStartState()
    if problem.isGoalState(start_node):
        return []

    visited = []
    my_stack = util.Stack()
    my_stack.push((start_node, []))

    reached_depth = 0

    while not my_stack.isEmpty():
        curr_node, actions = my_stack.pop()

        if curr_node not in visited:
            visited.append(curr_node)
 
            """ if the reached_depth is lower than limit, if current node is the goal, we return a sequence of
            actions, otherwise, if the limit was overcomed, a null sequence is returned, else if goalState is not found
            and limit not reached, the algorithm continues with visiting the successors of current node"""
            if problem.isGoalState(curr_node):
              return actions

            if reached_depth > depth:
                return []
            else:
                reached_depth += 1

                for succesor, next_action, cost in problem.getSuccessors(curr_node):
                    new_action = actions + [next_action]
                    my_stack.push((succesor, new_action))


"""The Depth First Limited Search algorithm is infinitely applied starting from minimum depth = 0 until the goalState
is reached, with incriseangly depth"""
def iterative_DLS(problem):
    depth = 0
    
    while True:
        actions = depthLimitedSearch(problem, depth)
        if len(actions) > 0:
            return actions
        else:
            depth += 1

"""in the second version of iiterative deepening deph first search, we summarise the process in a single function, 
by consecutively running depth first searches at increasingly levels of depth, marked here as trashholds;
the difference between the first and second implementation of IDDFS is that, here, we verify the depth we reached in the successors of current node for loop"""
def iterativeDeepeningSearch(problem):
    start_node = problem.getStartState()

    my_stack = util.Stack()
    my_stack.push((start_node, [], 0))

    trashhold = 0

    while not my_stack.isEmpty():
        trashhold += 1
        curr_node, actions, cost = my_stack.pop()
        visited = []

        visited.append(curr_node)
 
        while True:
            for succesor, next_action, next_cost in problem.getSuccessors(curr_node):
                if succesor not in visited and (cost + next_cost) <= trashhold:
                    visited.append(succesor)
                    my_stack.push((succesor, actions + [next_action], cost + next_cost))

            if my_stack.isEmpty():
                break

            curr_node, actions, cost = my_stack.pop()
            if problem.isGoalState(curr_node): 
                return actions

        my_stack.push((start_node, [], 0))

"""iterative deepening search star is an alternative of iterative deepening Depth First Search that borrows
the idea to use a heuristic function to evaluate the remaining cost to get to the goal from the A* search 
algorithms."""
def iterativeDeepeningSearchStar(problem, heuristic=nullHeuristic):
    start_node = problem.getStartState()

    my_stack = util.Stack()
    my_stack.push((start_node, [], 0))

    trashhold = 0

    while not my_stack.isEmpty():
        trashhold += 1
        curr_node, actions, cost = my_stack.pop()
        visited = []
        visited.append(curr_node)

        while True:
            for succesor, next_action, next_cost in problem.getSuccessors(curr_node):
                """the difference that interfeer is here, at the evauation of reaching the trashhold
                comparing this heuristic with the limit"""
                new_heuristic = cost + next_cost + heuristic(succesor, problem)
                if succesor not in visited and new_heuristic <= trashhold:
                    visited.append(succesor)
                    my_stack.push((succesor, actions + [next_action], cost + next_cost))

            if my_stack.isEmpty():
                break

            curr_node, actions, cost = my_stack.pop()
            if problem.isGoalState(curr_node): 
                return actions

        my_stack.push((start_node, [], 0))


def simulatedAnnealing(problem):
    curr_node = problem.getStartState() 
    actions = []

    """ the algorithm starts at temperature T=1 and controls the decreasingly change over time 
   and allows remaining in the same local maxima for max_iterations the most"""
    T = 1.0 
    max_iterations = 50

    while True:
        no_iterations = 0

        while no_iterations <= max_iterations:
            no_iterations += 1
            no_of_successors = 0
 
            """ in a Queue we push all the successors of the current node and 
            we keep track of the number of successors in order to choose a randomly one and to compare their
            values"""
            my_queue = util.Queue()
            for successor, directions, cost in problem.getSuccessors(curr_node):
                my_queue.push((successor, [directions], cost))
                no_of_successors += 1

            random_succ = random.randint(0, no_of_successors-1)
            for idx in range(0, random_succ+1):
                next_node, action, new_cost = my_queue.pop()

            """ is assigning the value of new variable to our current variable, we allow the assignment, 
            otherwise, we allow it only with the probability calculated odwn below"""
            E = new_cost - cost
            if E > 0:
                curr_node = next_node
                actions += action
            else:
                if math.exp(E/T):
                    curr_node = next_node
                    actions += action
        
            if problem.isGoalState(curr_node):
                return actions
        """the temperature is decreasingly, by being multiplied with a subunit factor, alfa, 0.8 < alfa < 0.995,
        in this case, alfa = 0.87"""
            
        T *= 0.87

"""in this algorithm, an efficient selection of the current best candidate for extension is tipically
implementing using a priority queue. in this way, the algorithm perform a selection of  minumim(estimated) 
cost nodes to expand. Greedy is a variant of best first search algorithm, along with a*,
using a different heuristic function"""
def greedyBestFirstSearch(problem, heuristic=nullHeuristic):
    start_node = problem.getStartState()
    visited = []

    if problem.isGoalState(start_node):
        return []

    my_priorityQ = util.PriorityQueue()
    my_priorityQ.push((start_node, [], 0), 0)

    while not my_priorityQ.isEmpty():
        curr_node, actions, p_cost = my_priorityQ.pop()

        if curr_node not in visited:
            visited.append(curr_node)

            if problem.isGoalState(curr_node):
                return actions

            for next, action, cost in problem.getSuccessors(curr_node):
                new_action = actions + [action]
                new_cost = p_cost + cost
                """greedy best first search expands the node that appears to be closest to goal, using 
                heuristic function f(n), f(n) = the estimated cost fromcurrent node n to reach the goal"""
                my_priorityQ.push((next, new_action, new_cost), heuristic(next, problem))

"""in this algorithm, an efficient selection of the current best candidate for extension is tipically
implementing using a priority queue. in this way, the algorithm perform a selection of  minumim(estimated) 
cost nodes to expand. 
Different from the greedy best first search, a* uses a better defined heuristic, which takes into
account both costs from initial state to node X and estimated cost from node X to goalState
Weighted A* expands states in order of f(X) = g(X) + W*h(X), where W = biass towards states 
that are closer to the goal"""
def aWeightedStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = []
    start_node = problem.getStartState()

    if problem.isGoalState(start_node):
        return []

	"nod, actiune, cost curent nod, prioritate"
    my_priorityQ = util.PriorityQueue()
    my_priorityQ.push((start_node, [], 0), 0)

    while not my_priorityQ.isEmpty():
        curr_node, actions, p_cost = my_priorityQ.pop()

        if curr_node not in visited:
            visited.append(curr_node)

            if problem.isGoalState(curr_node):
                return actions

            for next, action, cost in problem.getSuccessors(curr_node):
                new_action = actions + [action]
                new_cost = p_cost + cost
                new_heuristic = new_cost + 4.5*heuristic(next, problem)
                my_priorityQ.push((next, new_action, new_cost), new_heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
rs = randomSearch

ids = iterativeDeepeningSearch 
idls = iterative_DLS
idsstar = iterativeDeepeningSearchStar
sa = simulatedAnnealing
gbfs = greedyBestFirstSearch
awstar = aWeightedStarSearch













