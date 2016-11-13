import os
import json
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd


def printLine():
    print("#######################")


def printRow(row):
    print("##  " + str(row[0]) + "  ##  " + str(row[1]) + "  ##  " + str(row[2]) + "  ##")


def printBoard(state):
    printLine()
    for row in state:
        printRow(row)
        printLine()
    return


def translateAction(action):
    row = action // 3
    column = action % 3
    return row, column


def checkRow(state, row, symbol):
    for cell in state[row]:
        if (cell != symbol):
            return False
    return True


def checkColumn(state, column, symbol):
    for cell in [row[column] for row in state]:
        if (cell != symbol):
            return False
    return True

def hasWon(state, symbol):
    rightDiagonal = True
    leftDiagonal = True
    for i in range(3):
        if (checkRow(state, i, symbol)):
            return True
        if (checkColumn(state, i, symbol)):
            return True
        if (state[i][i] != symbol):
            rightDiagonal = False
        if (state[2-i][i] != symbol):
            leftDiagonal = False
    return rightDiagonal or leftDiagonal


class Game:

    def __init__(self):
        self.state = numpy.zeros((3, 3), dtype=numpy.int)

    def boardFull(self):
        for row in self.state:
            for cell in row:
                if (cell == 0):
                    return False
        return True

    def act(self, action, symbol):
        [row, column] = translateAction(action)
        self.state[row][column] = symbol
        reward = self.getReward(symbol)

        victory = -1
        if (hasWon(self.state, symbol)):
            victory = symbol
        elif (self.boardFull()):
            victory = 0

        return self.state.reshape(1, -1), reward, victory

    def isValidAction(self, action):
        [row, column] = translateAction(action)
        if (row > 2 or row < 0 or column > 2 or column < 0):
            return False
        return self.state[row][column] == 0

    def getFlatState(self):
        return self.state.reshape(1, -1)

    def reset(self):
        self.state = numpy.zeros((3, 3), dtype=numpy.int)

    def getReward(self, symbol):
        reward = 0
        if (hasWon(self.state, symbol)):
            reward += 100
        return reward


class ExperienceReplay:

    def __init__(self, maxMemory=100, discount=0.9):
        self.maxMemory = maxMemory
        self.memory = []
        self.discount = discount

    def remember(self, states, gameOver):
        self.memory.append([states, gameOver])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]

    def getBatch(self, model, batchSize=10):
        memoryLength = len(self.memory)
        numberOfActions = model.output_shape[-1]
        dimensions = self.memory[0][0][0].shape[1]
        inputs = numpy.zeros((min(memoryLength, batchSize), dimensions))
        targets = numpy.zeros((inputs.shape[0], numberOfActions))
        
        for i, item in enumerate(numpy.random.randint(0, memoryLength, size=inputs.shape[0])):
            previousState, action, reward, state = self.memory[item][0]
            gameOver = self.memory[item][1]
            inputs[i:i + 1] = previousState
            targets[i] = model.predict(previousState)[0]
            qValue = numpy.max(model.predict(state)[0])
            if (gameOver):
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * qValue

        return inputs, targets
    

def getConfig(filename):
    gridSize = 9
    config = {
            "exploration": 0.1,
            "gridSize": gridSize,
            "numberOfActions": gridSize,
            "epochs": 20,
            "maxMemory": 500,
            "hiddenSize": 100,
            "batchSize": 50,
            "filename": filename,
    }
    return config


def getModel(config):
    model = Sequential()
    model.add(Dense(config["hiddenSize"], input_shape=(9,), activation='relu'))
    model.add(Dense(config["hiddenSize"], activation='relu'))
    model.add(Dense(config["numberOfActions"]))
    model.compile(sgd(lr=0.2), "mse")

    weightsFilename = config["filename"] + ".h5"
    if (os.path.isfile(weightsFilename)):
        model.load_weights(weightsFilename)

    return model


class RandomPlayer:
    def __init__(self, config, symbol):
        self.symbol = symbol
        self.winCount = 0
        self.loss = 0.0

    def getAction(self, game, previousState):
        action = numpy.random.randint(0, 9)
        while (not game.isValidAction(action)):
            action = numpy.random.randint(0, 9)
        return action


class Player:
    def __init__(self, config, symbol):
        self.symbol = symbol
        self.winCount = 0
        self.loss = 0.0

    def getAction(self, game, previousState):
        printBoard(game.state)
        action = int(input("Please choose an action [0-8]:"))
        while (not game.isValidAction(action)):
            action = int(input("Not valid, please try again:"))
        return action


class Learner:
    def __init__(self, config, symbol):
        self.config = config
        self.symbol = symbol
        self.winCount = 0
        self.loss = 0.0
        self.model = getModel(self.config)
        self.experienceReplay = ExperienceReplay(maxMemory=config["maxMemory"])

    def getAction(self, game, previousState):
        action = None
        if (numpy.random.rand() <= self.config["exploration"]):
            action = numpy.random.randint(0, self.config["numberOfActions"])
            while (not game.isValidAction(action)):
                action = numpy.random.randint(0, self.config["numberOfActions"])
        else:
            q = self.model.predict(previousState).reshape(9,)
            action = numpy.argmax(q)
            while (not game.isValidAction(action)):
                q[action] = -10
                action = numpy.argmax(q)
        return action

    def train(self, previousState, action, reward, state, gameOver):
        self.experienceReplay.remember([previousState, action, reward, state], gameOver)
        inputs, targets = self.experienceReplay.getBatch(self.model, batchSize=self.config["batchSize"])
        self.loss += self.model.train_on_batch(inputs, targets)

    def save(self):
        filename = self.config["filename"]
        self.model.save_weights(filename + ".h5", overwrite=True)
        with open(filename + ".json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)



if __name__ == "__main__":
    brain = Learner(getConfig("brain"), 2)
    game = Game()

    tiedCount = 0
    answer = ''
    opponent = None

    while (answer != 'q'):
        answer = input("Choose an option: (q)uit, (p)lay, (t)rain")
        if (answer == 'q'):
            brain.save()
            if (hasattr(opponent, "save")):
                opponent.save()
            break;
        elif (answer == 'p'):
            opponent = Player(getConfig("Not applicable"), 1)
        elif (answer == 't'):
            opponent = Learner(getConfig("brain2"), 1)
        else:
            print('Please choose one of the options')
            continue;

        for epoch in range(brain.config["epochs"]):
            brain.loss = 0.0
            opponent.loss = 0.0

            game.reset()
            gameOver = False
            state = game.getFlatState()
            currentPlayer = brain
            brainIsPlaying = True

            while (not gameOver):
                previousState = state
                action = currentPlayer.getAction(game, previousState)
                state, reward, victory = game.act(action, currentPlayer.symbol)

                if (victory == currentPlayer.symbol):
                    currentPlayer.winCount += 1
                    gameOver = True

                if (victory == 0):
                    tiedCount += 1
                    gameOver = True

                if (hasattr(currentPlayer, "train")):
                    currentPlayer.train(previousState, action, reward, state, gameOver)

                currentPlayer = opponent if brainIsPlaying else brain
                brainIsPlaying = not brainIsPlaying

            print("Epoch {:03d}/999 | Loss {:.4f} | Win count {} | Tied count {} | Loss count {}".format(epoch, brain.loss, brain.winCount, tiedCount, opponent.winCount))


