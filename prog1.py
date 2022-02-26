import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Settings
START_CAPITAL = 1000    # Starting capital
DATASIZE = 1000         # Number of samples
MACD_PERIOD_1 = 26      # Longer period for MACD
MACD_PERIOD_2 = 12      # Shorter period for MACD
SIGNAL_PERIOD = 9       # Period for SIGNAL

offsetMACD = 0
offsetSIGNAL = 0

# assigning sample vector from .csv file
dataOverflowFlag = 0
df = pd.read_csv("wig20_d.csv", delimiter=",")
if len(df.index) < DATASIZE:
    DATASIZE = len(df.index)
    dataOverflowFlag = 1
csvData = df.tail(DATASIZE)
csvData = csvData.to_numpy()
csvData = csvData[:, 4]

# setting up arrays for data
arrayMACD = np.zeros(DATASIZE - MACD_PERIOD_1)
arraySIGNAL = np.zeros(DATASIZE - MACD_PERIOD_1 - SIGNAL_PERIOD)
arrayDIFF = np.zeros(DATASIZE - MACD_PERIOD_1 - SIGNAL_PERIOD)

# function calculating EMA
def ema(arr: np.array, N: int, offset: int):
    # declaring variables
    alfa = 2 / (N + 1)
    numerator = 0
    denominator = 0
    # calculating numerator and denominator
    for i in range(N, -1, -1):
        numerator = numerator + (1 - alfa) ** ((i - N) * -1) * arr[i + offset]
        denominator = denominator + (1 - alfa) ** ((i - N) * -1)
    return numerator / denominator


# function calculating MACD values
def macd(arr: np.array, N1: int, N2: int, offset: int):
    return ema(arr, N1, offset + N2 - N1) - ema(arr, N2, offset)


# function calculating MACD and SIGNAL datasets
def calcData(arrMA: np.array, arrSI: np.array, offMA: int, offSI: int, dataCSV: np.array):
    # calculating MACD dataset
    for i in range(0, DATASIZE - MACD_PERIOD_1):
        arrMA[i] = macd(dataCSV, MACD_PERIOD_2, MACD_PERIOD_1, offMA)
        offMA += 1
    # calculating SIGNAL from calculated MACD dataset
    for i in range(0, DATASIZE - MACD_PERIOD_1 - SIGNAL_PERIOD):
        arrSI[i] = ema(arrMA, SIGNAL_PERIOD, offSI)
        offSI += 1


def calcDiff(arrMA: np.array, arrSI: np.array, arrDI: np.array):
    for i in range(0, arrSI.size):
        arrDI[i] = arrMA[i + SIGNAL_PERIOD] - arrSI[i]


# creating graphs
# creating MACD graph
def drawMACD_SIGNAL():
    instancesMACD = np.arange(DATASIZE - MACD_PERIOD_1)
    instancesSIGNAL = np.arange(SIGNAL_PERIOD, DATASIZE - MACD_PERIOD_1)
    plt.plot(instancesMACD, arrayMACD, color="tab:blue", linewidth="1")                             # MACD
    plt.plot(instancesSIGNAL, arraySIGNAL, color="tab:red", linewidth="1")                          # SIGNAL
    plt.xlim(0, DATASIZE)
    plt.ylabel("MACD value")
    plt.xlabel("Days")
    plt.title("MACD Indicator")
    plt.legend(["MACD", "SIGNAL"])
    plt.grid()
    plt.show()

#creating WIG20 graph
def drawValues():
    plt.plot(np.arange(0, DATASIZE), csvData, color='tab:green', linewidth='1')                             # WIG20
    plt.xlim(0, DATASIZE)
    plt.ylabel("Closing price")
    plt.xlabel("Days")
    plt.title("Values of WIG20")
    plt.legend(["WIG20"])
    plt.grid()
    plt.show()


def debugValues(arrMACD, arrSIGNAL, arrDIFF):
    for i in range(0, arrayDIFF.size):
        print(
            "Sample: "
            + str(i + 9)
            + " MACD: "
            + str(arrMACD[i + SIGNAL_PERIOD])
            + " SIGNAL: "
            + str(arrSIGNAL[i])
            + " DIFF: "
            + str(arrDIFF[i])
        )
    return


# Trading algorithms
# (Assuming fractional shares)
# 1. Buy when diff changes from     - to +
#    Sell when diff changes from    + to -
def tradingAlg(arrDI, arrCSV, capital):
    print("Starting capital: " + str(capital))
    print("\n" + "#####TRADING#####" + "\n")
    ownedShares = 0
    lastCapital = 0
    prevFlag = 1 if arrDI[0] >= 0 else 0
    for i in range(1, arrDI.size):
        currentFlag = 1 if arrDI[i] >= 0 else 0
        if prevFlag ^ currentFlag == 1:
            if prevFlag == 1:  # currentFlag == 0
                # sell
                print("\nSell")
                print(
                    "Sample: "
                    + str(i + SIGNAL_PERIOD)
                    + " Owned shares: "
                    + str(ownedShares)
                    + " Current Price: "
                    + str(arrCSV[SIGNAL_PERIOD + MACD_PERIOD_1 + i])
                )
                capital += ownedShares * arrCSV[SIGNAL_PERIOD + MACD_PERIOD_1 + i]
                lastCapital = capital
                print("Capital: " + str(round(capital, 2)))
                ownedShares = 0
            else:  # currentFlag == 1
                # buy
                print("\nBuy")
                print(
                    "Sample: "
                    + str(i + SIGNAL_PERIOD)
                    + " Capital: "
                    + str(round(capital, 2))
                    + " Current Price: "
                    + str(arrCSV[SIGNAL_PERIOD + MACD_PERIOD_1 + i])
                )
                ownedShares += capital / arrCSV[SIGNAL_PERIOD + MACD_PERIOD_1 + i]
                print("Owned shares: " + str(ownedShares))
                capital = 0
        prevFlag = currentFlag
    if capital == 0:
        # case when last operation was buy
        print("Capital: " + str(capital))
        print("Owned shares: " + str(ownedShares))
        print(
            "Capital if shares were sold at starting price: "
            + str(round(ownedShares * arrCSV[SIGNAL_PERIOD + MACD_PERIOD_1], 2))
        )
        print("Capital after last sell operation: " + str(round(lastCapital, 2)))
    else:
        # case when last operation is sell
        print("\nEnd of trading algorithm")
        print("Capital: " + str(round(capital, 2)))
    return


def main():
    # operations on data
    calcData(arrayMACD, arraySIGNAL, offsetMACD, offsetSIGNAL, csvData)
    calcDiff(arrayMACD, arraySIGNAL, arrayDIFF)
    tradingAlg(arrayDIFF, csvData, START_CAPITAL)
    if dataOverflowFlag == 1:
        print(
            "Tried to access more entries than .csv file contains.\nChanged DATASIZE to match number of entries in .csv file."
        )
        print("DATASIZE =", DATASIZE)
    drawMACD_SIGNAL()
    drawValues()


main()