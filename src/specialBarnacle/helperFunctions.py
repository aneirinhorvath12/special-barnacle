import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def displayFigure(displayFormat, fileName=''):
    if (displayFormat == "figure"):
        plt.show()
    elif (displayFormat == "png"):
        plt.savefig(fileName + '.png')
    else:
        plt.savefig(fileName + '.pdf') 

def addIndexColumnToDF(df):
    dataFrameSize = df.shape[0]
    indexColumn = range(0, dataFrameSize)
    df['Index'] = indexColumn
    return df

def readCSVs():
    repeat = True
    logsArrays = []
    while repeat:
        logFile = input('Enter log file name. Enter END to end. ')
        if logFile == 'END':
            repeat = False
        else: 
            df = pd.read_csv(logFile)
            logsArrays.append(df)
            continue
    return logsArrays

def calculateNoClusters(df, maxK):
    means = []
    inertias = []

    for k in range(1, maxK):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        means.append(k)
        inertias.append(kmeans.inertia_)
    
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.grid(True)
    plt.show()

def adjustColumnHeadings(columnHeadings):
    columnHeadersAdjusted = []
    for heading in columnHeadings:
        temp = heading + '_T'
        columnHeadersAdjusted.append(temp)
    return columnHeadersAdjusted
    