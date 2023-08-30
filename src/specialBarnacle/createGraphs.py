import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
import helperFunctions as hf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def createBoxPlot(dataSet, figureHeight, figureWidth, title, xAxisLabel, yAxisLabel, hasGrid, displayFormat, fileName="", groupBy=""):
    df = pd.read_csv(dataSet)
    df.dropna(inplace=True)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y=yAxisLabel, x=xAxisLabel, hue=groupBy, ax=ax)
    plt.rcParams['figure.figsize']=[figureWidth, figureHeight]
    plt.title(title)
    ax.yaxis.grid(hasGrid)
    ax.xaxis.grid(hasGrid)
    hf.displayFigure(displayFormat, fileName)

def createGraphs(logsArraySquared, confidenceLevel, figureHeight, figureWidth, hasGridlines, xAxisLabel, yAxisLabel, title, displayFormat, fileName=""):
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize']=[figureWidth, figureHeight]
    plt.title(title)
    ax.yaxis.grid(hasGridlines)
    ax.xaxis.grid(hasGridlines)
    indexedLogsArraySquared = []
    for logArray in logsArraySquared:
        indexedLogsArray = []
        for log in logArray:   
            df = pd.read_csv(log)
            df.dropna(inplace=True)
            dfIndexed = hf.addIndexColumnToDF(df)
            indexedLogsArray.append(dfIndexed)
        indexedLogsArraySquared.append(indexedLogsArray)
    for logArray in indexedLogsArraySquared:
        dataSet = pd.concat(logArray, ignore_index=True)
        sns.lineplot(dataSet, x=xAxisLabel, y=yAxisLabel, errorbar=('ci', confidenceLevel))
    hf.displayFigure(displayFormat, fileName="")

def createLiveGraph(logs, figureHeight, figureWidth, xAxisLabel, yAxisLabel, title, displayFormat, fileName=''):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    plt.rcParams['figure.figsize']=[figureWidth, figureHeight]
    def animate(self):
        dataSet = pd.read_csv(logs)
        dataSet.dropna(inplace=True)
        x = dataSet[xAxisLabel].to_numpy()
        y = dataSet[yAxisLabel].to_numpy()
        ax.clear()
        ax.plot(x, y)
    anim = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
    plt.show()

# Currently only using K-Means clustering - future improvements include using more clustering methods?
def clusteredGraph(isAlreadyClustered, dataSet, xAxisLabel, yAxisLabel, noClusters, figureWidth, figureHeight, title, displayFormat, zAxisLabel='', fileName=''):
    df = pd.read_csv(dataSet)
    fig = plt.figure(figsize = (16, 9))
    plt.title(title)
    if zAxisLabel == '':
        ax = plt.axes()
    else:
        ax = plt.axes(projection ="3d")
        ax.set_zlabel(zAxisLabel)
    ax.set_xlabel(xAxisLabel)
    ax.set_ylabel(yAxisLabel)
    if (not isAlreadyClustered): 
        df.dropna(inplace=True)
        columnHeaders = list(df.columns.values)
        columnHeadersAdjusted = hf.adjustColumnHeadings(columnHeaders)
        xAxisLabelAdjusted = xAxisLabel + '_T'
        yAxisLabelAdjusted = yAxisLabel + '_T'
        zAxisLabelAdjusted = zAxisLabel + '_T' 
        # Standardize the data
        scaler = StandardScaler()
        df[columnHeadersAdjusted] = scaler.fit_transform(df[columnHeaders])
        # Perform clustering
        kmeans = KMeans(noClusters)
        if zAxisLabel == '':
            kmeans.fit(df[[xAxisLabelAdjusted, yAxisLabelAdjusted]])
        else:
            kmeans.fit(df[[xAxisLabelAdjusted, yAxisLabelAdjusted, zAxisLabelAdjusted]])
        df['kmeans_cluster'] = kmeans.labels_
    if zAxisLabel == '':
        plt.scatter(x=df[yAxisLabel], y=df[xAxisLabel], c=df['kmeans_cluster'])
    else:
        ax.scatter(xs=df[yAxisLabel], ys=df[xAxisLabel], zs=df[zAxisLabel], c=df['kmeans_cluster'])
    plt.xlim(-0.1, 1)
    plt.ylim(3, 1.5)
    hf.displayFigure(displayFormat, fileName="")


def highDimensionalGraph(data, target, targetNames, noOfDimensions, title, displayFormat, fileName=''):
    # Creating figure
    fig = plt.figure(figsize = (16, 9))
    if noOfDimensions == 2:
        ax = plt.axes()
    else:
        ax = plt.axes(projection ="3d")
    scaler = StandardScaler()
    xScaled = scaler.fit_transform(data)

    pca = PCA(n_components=noOfDimensions)
    pca_features = pca.fit_transform(xScaled)
    if noOfDimensions == 2:
        pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
    else:
        pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2', 'PC3'])
    pca_df['target'] = target
    if noOfDimensions == 2:
        ax.scatter(x=pca_df['PC1'], y=pca_df['PC2'], c=pca_df['target'])
    else:
        ax.scatter(xs=pca_df['PC1'], ys=pca_df['PC2'], zs=pca_df['PC3'], c=pca_df['target'])
    pca_df['target'] = pca_df['target'].map(targetNames)
    plt.legend(labels=pca_df['target'])
    plt.title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if noOfDimensions == 3:
        ax.set_zlabel('PC3')
    hf.displayFigure(displayFormat, fileName="")

# A scree plot to determine the number of PCs for a dataset 
def screePlot(data):
    scaler = StandardScaler()
    xScaled = scaler.fit_transform(data)

    pca = PCA(n_components=4)
    pca_features = pca.fit_transform(xScaled)
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eignevalues')
    plt.show()

def calculateNoClusters(dataSet, xAxisLabel, yAxisLabel):
    df = pd.read_csv(dataSet)
    df.dropna(inplace=True)
    columnHeaders = list(df.columns.values)
    columnHeadersAdjusted = hf.adjustColumnHeadings(columnHeaders)
    xAxisLabelAdjusted = xAxisLabel + '_T'
    yAxisLabelAdjusted = yAxisLabel + '_T'
    # Standardize the data
    scaler = StandardScaler()
    df[columnHeadersAdjusted] = scaler.fit_transform(df[columnHeaders])
    # Identify optimal no. of clusters using Elbow method
    hf.calculateNoClusters(df[[xAxisLabelAdjusted, yAxisLabelAdjusted]], 10)
