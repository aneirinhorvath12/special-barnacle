import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def createBoxPlot(dataSet, figureHeight, figureWidth, title, xAxisLabel, yAxisLabel, hasGrid, displayFormat, fileName=""):
    fig, ax = plt.subplots(figsize=(figureHeight, figureWidth)) # Set figure height & width
    dataSet.boxplot(by=xAxisLabel, column=yAxisLabel, grid=hasGrid)
    ax.set_ylabel(yAxisLabel) # Add y axis label 
    ax.legend()
    if (displayFormat == "figure"):
        plt.show()
    elif (displayFormat == "png"):
        plt.savefig(fileName + '.png')
    else:
        plt.savefig(fileName + '.pdf') 

def createGraphs(logsArray, confidenceLevel, figureHeight, figureWidth, hasGridlines, xAxisLabel, yAxisLabel, isLog, displayFormat, fileName=""):
    indexedLogsArray = []
    for log in logsArray:
        df = addIndexColumnToDF(log)
        indexedLogsArray.append(df)
    dataSet = pd.concat(indexedLogsArray, ignore_index=True)
    ax = sns.lineplot(dataSet, x="Index", y="r", errorbar=('ci', 95))
    ax.plot()
    plt.show()
    """
    # Plot the results of the experiments 
    for column in dataSet:
        # Assumption: the x-axis for each experiment will be called row (could be improved by making this more dynamic)
        if (column != 'r'):
            ax.plot(dataSet['r'], dataSet[column], label=column)
    if (displayFormat == "figure"):
        plt.show()
    elif (displayFormat == "png"):
        plt.savefig(fileName + '.png')
    else:
        plt.savefig(fileName + '.pdf') 
    """

def addIndexColumnToDF(df):
    dataFrameSize = df.shape[0]
    indexColumn = range(0, dataFrameSize)
    df['Index'] = indexColumn
    return df

