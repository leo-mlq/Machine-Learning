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
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # print("legal moves: ", legalMoves);
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print("action taken: ", legalMoves[chosenIndex])
        # print('\n')
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print("new positions: ", newPos);
        # print("new food: ",newFood);
        # print("new ghost states: ", newGhostStates[0].getPosition());
        # print("new scared times: ", newScaredTimes);
        # print("successor game state score: ", successorGameState.getScore())
        "*** YOUR CODE HERE ***"

        score = 0;
        for scaredTime in newScaredTimes:
            score+=scaredTime+18

        score+=successorGameState.getScore()

        ghostDis=[manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        nearestGhostDis = min(ghostDis)
        if(len(newFood)==0):
            nearestFoodDis=0
        else:
            nearestFoodDis=9/min([manhattanDistance(newPos, food) for food in newFood])

        if(nearestGhostDis < 3):
            penalty = 18
        else:
            penalty = 0;

            if (len(newFood) < currentGameState.getNumFood()):
                score+=18

            if newPos in currentGameState.getCapsules():
                score+=18

        if action == Directions.STOP:
            return -9999999;

       

        return score-currentGameState.getScore()+nearestFoodDis-penalty;

def scoreEvaluationFunction(currentGameState):
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
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

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
    Your minimax agent (question 7)
    """

    def getAction(self, gameState):
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
        """
        # print("pacman legal actions: ", gameState.getLegalActions(0))
        # print("number of ghosts: ", gameState.getNumAgents());
        # print("ghosts legal actions: ", gameState.getLegalActions(1))
        "*** YOUR CODE HERE ***"

        maxEval = float('-Inf')
        maxAction = ''
        for action in gameState.getLegalActions(0):
            
            # print("action: ", action)
            successorGameState = gameState.generateSuccessor(0,action)
            if gameState.getNumAgents()-1==0:
                newEval = self.miniMax(successorGameState, 0, 0)
            else:
                newEval = self.miniMax(successorGameState, 0, 1)
            # print("score: ",newEval)

            if(newEval > maxEval):
                maxEval = newEval
                maxAction = action
        # print("action taken: ",maxAction)
                # print("\n")
        return maxAction

    
    def miniMax(self, gameState, depth, agentIndex):
        if(depth == self.depth) or gameState.isWin() or gameState.isLose():
            # print("this is end of game. depth is ", depth)

            return self.evaluationFunction(gameState);

        #pacman
        if(agentIndex == 0):
            if(gameState.getNumAgents()-1==0):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth  

            # print("pacman's turn and next depth is: ", nextDepth)
            maxEval = float('-Inf')
            pacmanActions = gameState.getLegalActions(0)
            for action in pacmanActions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                newEval = self.miniMax(successorGameState, nextDepth, nextAgent)
                # print("max score: ", newEval)
                maxEval = max(maxEval, newEval)
            
            return maxEval

        #ghosts
        else:
            #last ghost
            if(agentIndex == gameState.getNumAgents()-1):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth

            # print("agent: ", agentIndex, " depth: ", depth)
            minEval = float('Inf')
            ghostActions = gameState.getLegalActions(agentIndex)
            for action in ghostActions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                newEval = self.miniMax(successorGameState, nextDepth, nextAgent)
                # print("min score: ", newEval)
                minEval = min(minEval, newEval)
            
            return minEval


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxEval = float('-Inf')
        maxAction = ''
        for action in gameState.getLegalActions(0):
            
            # print("action: ", action)
            successorGameState = gameState.generateSuccessor(0,action)
            if gameState.getNumAgents()-1==0:
                newEval = self.expect(successorGameState, 0, 0)
            else:
                newEval = self.expect(successorGameState, 0, 1)
            # print("score: ",newEval)

            if(newEval > maxEval):
                maxEval = newEval
                maxAction = action
        # print("action taken: ",maxAction)
                # print("\n")
        return maxAction

    
    def expect(self, gameState, depth, agentIndex):
        if(depth == self.depth) or gameState.isWin() or gameState.isLose():
            # print("this is end of game. depth is ", depth)

            return self.evaluationFunction(gameState);

        #pacman
        if(agentIndex == 0):
            if(gameState.getNumAgents()-1==0):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth  

            # print("pacman's turn and next depth is: ", nextDepth)
            maxEval = float('-Inf')
            pacmanActions = gameState.getLegalActions(0)
            for action in pacmanActions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                newEval = self.expect(successorGameState, nextDepth, nextAgent)
                # print("max score: ", newEval)
                maxEval = max(maxEval, newEval)
            
            return maxEval

        #ghosts
        else:
            #last ghost
            if(agentIndex == gameState.getNumAgents()-1):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth

            # print("agent: ", agentIndex, " depth: ", depth)
            minEval = float('Inf')
            ghostActions = gameState.getLegalActions(agentIndex)
            expectScore = []
            for action in ghostActions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                newEval = self.expect(successorGameState, nextDepth, nextAgent)
                # print("min score: ", newEval)
                expectScore.append(newEval)

            
            
            return sum(expectScore)/len(expectScore)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # print("new positions: ", newPos);
    # print("new food: ",newFood);
    # print("new ghost states: ", newGhostStates[0].getPosition());
    # print("new scared times: ", newScaredTimes);
    # print("successor game state score: ", successorGameState.getScore())
    "*** YOUR CODE HERE ***"

    score = 0;
    score+=currentGameState.getScore()
    ghostDis=[manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    if(len(ghostDis)==0):
        nearestGhostDis=999999999999999
    else:
        nearestGhostDis = min(ghostDis)

    capsulesDis = [manhattanDistance(newPos, caps) for caps in currentGameState.getCapsules()]
    if(len(capsulesDis)==0):
        nearestCapsule=0
    else:
        nearestCapsule=9/min(capsulesDis)

    if(len(newFood)==0):
        nearestFoodDis=0
    else:
        nearestFoodDis=9/min([manhattanDistance(newPos, food) for food in newFood])

    # distance of 3 to ghost will keep pacman safe
    if(nearestGhostDis < 3):
        # when distance to ghost less than 3, add penalty to cancel out attraction caused by nearest food. 
        # if nearest food is 1, 9/1 + 9 (get points if eat it) = 18
        penalty = 18
        # if pacman ate a capsule, try to catch a nearest ghost by rewarding 501 (a number by experiment)
        for scaredTime in newScaredTimes:
            score+=scaredTime+501/len(newScaredTimes)

    #if distance to ghost is at least three.
    else:
        #no penalty, pacman can move freely
        penalty = 0;
        #try to get the nearest food or capsule by giving it a larger score
        score += max(nearestFoodDis,nearestCapsule)+501
        
    return score-penalty;
    

# Abbreviation
better = betterEvaluationFunction

