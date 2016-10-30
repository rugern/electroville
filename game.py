state = [[0 for i in range(3)] for j in range(3)]


def boardFull():
    for row in state:
        for cell in row:
            if (cell == 0):
                return False
    return True


def printLine():
    print("#######################")


def printRow(row):
    print("##  " + str(row[0]) + "  ##  " + str(row[1]) + "  ##  " + str(row[2]) + "  ##")


def printBoard():
    printLine()
    for row in state:
        printRow(row)
        printLine()
    return


def performMove(move, symbol):
    [row, column] = move
    state[row][column] = symbol


def checkRow(row, symbol):
    for cell in state[row]:
        if (cell != symbol):
            return False
    return True


def checkColumn(column, symbol):
    for cell in [row[column] for row in state]:
        if (cell != symbol):
            return False
    return True


def hasWon(symbol):
    diagonal = True
    for i in range(3):
        if (checkRow(i, symbol)):
            return True
        if (checkColumn(i, symbol)):
            return True
        if (state[i][i] != symbol):
            diagonal = False

    return diagonal

def isValidMove(move):
    [row, column] = move
    if (row > 2 or row < 0 or column > 2 or column < 0):
        return False
    return state[row][column] == 0


def getState():
    return state
