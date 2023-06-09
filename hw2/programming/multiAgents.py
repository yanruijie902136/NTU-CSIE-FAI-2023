# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Stopping reduces our scoring, so don't stop.
        if action == 'Stop':
            return -math.inf

        # Manhattan distance between agent and the ghost.
        ghostDist = manhattanDistance(newPos, newGhostStates[0].configuration.pos)

        newFoodList = newFood.asList()
        # If all foods have been eaten, this is the goal state, whose evaluation should be the highest.
        if len(newFoodList) == 0:
            return math.inf
        # Manhattan distance between agent and the closest food.
        closestFoodDist = math.inf
        for food in newFoodList:
            closestFoodDist = min(closestFoodDist, manhattanDistance(newPos, food))

        # Final score. The further the ghost is or the closer the closest food is, the better this state is.
        return successorGameState.getScore() + ghostDist / closestFoodDist


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getValue(self, gameState, agentIndex, depth):
        # Terminal states.
        # It is worth noting that the `depth` parameter doesn't have the same meaning as self.depth.
        # Instead, it's the depth in the minimax tree. Therefore, the terminal states at the bottom
        # of the minimax tree will satisfy
        #       depth == self.depth * gameState.getNumAgents()
        # rather than
        #       depth == self.depth.
        if depth == self.depth * gameState.getNumAgents() \
                or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            # Pacman.
            return self.maxValue(gameState, agentIndex, depth)[0]
        else:
            # Ghost(s).
            return self.minValue(gameState, agentIndex, depth)[0]


    def minValue(self, gameState, agentIndex, depth):
        minVal, optimalAction = math.inf, ""

        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            nextVal = self.getValue(nextGameState, nextAgentIndex, depth + 1)
            if nextVal < minVal:
                minVal, optimalAction = nextVal, action

        return minVal, optimalAction


    def maxValue(self, gameState, agentIndex, depth):
        maxVal, optimalAction = -math.inf, ""

        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            nextVal = self.getValue(nextGameState, nextAgentIndex, depth + 1)
            if nextVal > maxVal:
                maxVal, optimalAction = nextVal, action

        return maxVal, optimalAction


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # We start with the pacman, so we shall maximize the value.
        return self.maxValue(gameState, 0, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getValue(self, gameState, agentIndex, depth, alpha, beta):
        # Terminal states.
        if depth == self.depth * gameState.getNumAgents() \
                or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            # Pacman.
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)[0]
        else:
            # Ghost(s).
            return self.minValue(gameState, agentIndex, depth, alpha, beta)[0]


    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        minVal, optimalAction = math.inf, ""

        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            nextVal = self.getValue(nextGameState, nextAgentIndex, depth + 1, alpha, beta)
            if nextVal < minVal:
                minVal, optimalAction = nextVal, action

            # Alpha-beta pruning.
            if minVal < alpha:
                break
            beta = min(beta, minVal)

        return minVal, optimalAction


    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        maxVal, optimalAction = -math.inf, ""

        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            nextVal = self.getValue(nextGameState, nextAgentIndex, depth + 1, alpha, beta)
            if nextVal > maxVal:
                maxVal, optimalAction = nextVal, action

            # Alpha-beta pruning.
            if maxVal > beta:
                break
            alpha = max(alpha, maxVal)

        return maxVal, optimalAction


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # We start with pacman, so we shall maximize the value.
        return self.maxValue(gameState, 0, 0, -math.inf, math.inf)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getValue(self, gameState, agentIndex, depth):
        # Terminal states.
        if depth == self.depth * gameState.getNumAgents() \
                or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            # Pacman.
            return self.maxValue(gameState, agentIndex, depth)[0]
        else:
            # Ghost(s).
            return self.expValue(gameState, agentIndex, depth)[0]


    def maxValue(self, gameState, agentIndex, depth):
        maxVal, optimalAction = -math.inf, ""

        for action in gameState.getLegalActions(agentIndex):
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            nextVal = self.getValue(nextGameState, nextAgentIndex, depth + 1)
            if nextVal > maxVal:
                maxVal, optimalAction = nextVal, action

        return maxVal, optimalAction


    def expValue(self, gameState, agentIndex, depth):
        expVal = 0

        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            nextGameState = gameState.generateSuccessor(agentIndex, action)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

            expVal += self.getValue(nextGameState, nextAgentIndex, depth + 1) / len(legalActions)

        return expVal, ""


    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # We start with pacman, so we shall maximize the value.
        return self.maxValue(gameState, 0, 0)[1]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    The evaluation function I devised consider not only the manhattan distance to the closest food
    and ghost, but also the number of food, ghosts, and capsules. Initially, I've tried

            1 / foodCount + closestGhostDist + 1 / (closestFoodDist + 1) + 1 / (capsuleCount + 1)

    but the performance wasn't ideal. Then I tried to multiply different weights to different
    variables, and the performance was much better. Eventually I settled for

            500000 / foodCount + closestGhostDist + 500 / (closestFoodDist + 1) + 5000 / (capsuleCount + 1)

    since this evaluation is good enough for me to get all 6 points, with an average score of 1083.2.
    It's worth noting that both closestFoodDist and capsuleCount can be zero sometimes, so I added
    one to the denominators to avoid zero division.
    """
    "*** YOUR CODE HERE ***"

    currPos = currentGameState.getPacmanPosition()
    currFoodList = currentGameState.getFood().asList()
    currGhostStates = currentGameState.getGhostStates()
    currCapsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return math.inf
    elif currentGameState.isLose():
        return -math.inf

    # Manhattan distance to the closest food.
    closestFoodDist = math.inf
    for food in currFoodList:
        closestFoodDist = min(closestFoodDist, manhattanDistance(currPos, food))

    # Number of food.
    foodCount = len(currFoodList)
    # If there's no food, it's over.
    if foodCount == 0:
        return math.inf

    # Manhattan distance to the closest ghost.
    closestGhostDist = math.inf
    for ghostState in currGhostStates:
        closestGhostDist = min(closestGhostDist, manhattanDistance(currPos, ghostState.configuration.pos))

    # Number of ghosts.
    # ghostCount = len(currGhostStates)

    # Manhattan distance to the closest capsule.
    # closestCapsuleDist = math.inf
    # for capsule in currCapsules:
    #     closestCapsuleDist = min(closestCapsuleDist, manhattanDistance(currPos, capsule))

    # Number of capsules.
    capsuleCount = len(currCapsules)

    return 500000 / foodCount + closestGhostDist + 500 / (closestFoodDist + 1) + 5000 / (capsuleCount + 1)


# Abbreviation
better = betterEvaluationFunction
