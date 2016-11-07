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
    diagonal = True
    for i in range(3):
        if (checkRow(state, i, symbol)):
            return True
        if (checkColumn(state, i, symbol)):
            return True
        if (state[i][i] != symbol):
            diagonal = False
    return diagonal


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
        return self.state.reshape(1, -1), reward

    def isValidAction(self, action):
        [row, column] = translateAction(action)
        if (row > 2 or row < 0 or column > 2 or column < 0):
            return False
        return self.state[row][column] == 0

    def getState(self):
        return self.state.reshape(1, -1)

    def reset(self):
        self.state = numpy.zeros((3, 3), dtype=numpy.int)

    def getReward(self, symbol):
        reward = 0
        if (hasWon(self.getState(), symbol)):
            reward += 100
        return reward


def play(model):
    global playerTurn
    winner = 0
    while (not game.boardFull()):
        symbol = 1 if playerTurn else 2
        game.printBoard()
        move = getInput(model)
        while (not game.isValidMove(move)):
            move = getInput(model)

        game.act(move, symbol)
        if (hasWon(game.getState(), symbol)):
            return symbol
        playerTurn = not playerTurn
        print(playerTurn)
    return winner


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
    

def playerMove(game, symbol):
    action = numpy.random.randint(0, 9)
    while (not game.isValidAction(action)):
        action = numpy.random.randint(0, 9)
    game.act(action, symbol)


if __name__ == "__main__":
    exploration = 0.1
    gridSize = 9
    numberOfActions = gridSize
    epochs = 1000
    maxMemory = 500
    hiddenSize = 100
    batchSize = 50
    
    model = Sequential()
    model.add(Dense(hiddenSize, input_shape=(9,), activation='relu'))
    model.add(Dense(hiddenSize, activation='relu'))
    model.add(Dense(numberOfActions))
    model.compile(sgd(lr=0.2), "mse")

    game = Game()
    experienceReplay = ExperienceReplay(maxMemory=maxMemory)

    winCount = 0
    for epoch in range(epochs):
        loss = 0.0
        game.reset()
        gameOver = False
        state = game.getState()

        while (not gameOver):
            previousState = state
            action = None
            if (numpy.random.rand() <= exploration):
                action = numpy.random.randint(0, numberOfActions)
                while (not game.isValidAction(action)):
                    action = numpy.random.randint(0, numberOfActions)
            else:
                q = model.predict(previousState)
                action = numpy.argmax(q[0])
                while (not game.isValidAction(action)):
                    numpy.delete(q[0], action)
                    action = numpy.argmax(q[0])

            state, reward = game.act(action, 2)

            if (hasWon(game.getState(), 2)):
                winCount += 1
                gameOver = True
                break
                
            playerMove(game, 1)
            if (hasWon(game.getState(), 1)):
                print("Player won")
                gameOver = True
                break
            
            experienceReplay.remember([previousState, action, reward, state], gameOver)
            inputs, targets = experienceReplay.getBatch(model, batchSize=batchSize)
            loss += model.train_on_batch(inputs, targets)
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(epoch, loss, winCount))

    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

