import inquirer
import math
import pandas
import numpy
import gdal
import georasters as gr
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
sns.set(color_codes=True)
from sklearn import preprocessing
from numpy import linalg as LA


# dictionary holding filename and according element name
names = {'LP_GRS_Th_Global_2ppd.tif': 'Thorium',
    'LP_GRS_Fe_Global_2ppd.tif': 'Iron',
        'LP_GRS_H_Global_2ppd.tif' : 'Hydrogen'
    }

## FUNCTIONS

def plotHistogram(df):
    fig = plt.figure(figsize=(10,7))
    plt.xlabel('Value', fontsize= 20)
    plt.ylabel('Frequency', fontsize= 20)
    plt.title(df.columns.values.tolist()[-1], fontsize= 30)
    ax = fig.add_subplot(111)
    df = df.iloc[:,2].copy()
    ax = df.hist(figsize=(10,7), bins= 25)
    plt.show()

def plot(df, xname, yname):
    ax = df.plot(kind='scatter', x=xname, y=yname)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.suptitle('scatter plot', fontsize= 30)
    plt.show()

# input file (tif or xyz) -> output pandas dataframe
def fileToDataframe(file):
    df = gr.from_file(file).to_pandas()
    df = df[["x", "y", "value"]].copy()
    df.columns = ["x", "y", names[file]]
    return df


def aggregateValues(listOfDataframes):
    for index, df in enumerate(listOfDataframes):
        if index == 0:
            valuesDf = df.iloc[:,2].copy()
            valuesDf = valuesDf.to_frame()
        else:
            valuesDf = pandas.concat([valuesDf, df.iloc[:,2].copy()], axis=1)
    return valuesDf


# input list of dataframes -> output dataframe of mean, std, min, 25%, 50%, 75%, and max
def getStats(listofDataframes):
    valuesDf = aggregateValues(listofDataframes)
    return valuesDf.describe()



# input list of dataframes -> output dataframe which is a correlation matrix of dataframes passed
def getCorrelation(listOfDataframes):
    valuesDf = aggregateValues(listOfDataframes)
    return valuesDf.corr()

def visualizeCorrelation(listOfDataframes):
    correlations = getCorrelation(listOfDataframes)
#    print(correlations)
    fig = plt.figure(figsize=(10,7))
    plt.title("Correlation Matrix", fontsize= 30)
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, cmap=cm.gist_earth,alpha=0.7,vmin=correlations.min().min(), vmax=correlations.max().max())
    for (i, j), z in numpy.ndenumerate(correlations):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center')
    fig.colorbar(cax)
    ticks = numpy.arange(0,len(listOfDataframes),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(correlations.columns)
    ax.set_yticklabels(correlations.columns)
    plt.show()

def getMin(df):
    min = df.min().min()
    return min


def getMax(df):
    max = df.max().max()
    return max

def visualizeCovariance(listOfDataframes, norm = False):
    valuesDf = aggregateValues(listOfDataframes)
    print(norm)
    if norm:
        valuesDF = pandas.DataFrame(preprocessing.MinMaxScaler().fit_transform(valuesDf), columns=valuesDf.columns, index=valuesDf.index)
        print(valuesDf.head())
    covariance = valuesDf.cov()
    max = getMax(covariance)
    min = getMin(covariance)
    fig = plt.figure(figsize=(10,7))
    plt.title("Covariance Matrix", fontsize= 30)
    ax = fig.add_subplot(111)
    cax = ax.matshow(covariance, cmap=cm.gist_earth, alpha=0.7, vmin=covariance.min().min(), vmax=covariance.max().max())
    for (i, j), z in numpy.ndenumerate(covariance):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center')
    fig.colorbar(cax)
    ticks = numpy.arange(0,len(listOfDataframes),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(covariance.columns)
    ax.set_yticklabels(covariance.columns)
    plt.show()


## END OF FUNCTIONS

questions = [
             inquirer.Checkbox('Layers',
                               message="What layers do you want to analyze?",
                               choices=['LP_GRS_Th_Global_2ppd.tif', 'LP_GRS_Fe_Global_2ppd.tif', 'LP_GRS_H_Global_2ppd.tif'],
                               ),
             ]
answers = inquirer.prompt(questions)

df = []
# answers holds the layers to be analyzed
for answer in answers['Layers']:
    if answer in names:
        df.append(fileToDataframe(answer))
#df now is a list of dataframes for the files selected


if len(df) == 2:
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Distribution of Covariance Matrices', 'Scatter Plot']
# chose multiple layers
elif len(df) > 1:
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Distribution of Covariance Matrices']
# didn't choose any
elif len(df) == 0:
    print("You didn't choose any layers. Exiting.")
    exit(0)
# they chose 1 layer
else:
    choicesList = ['Stats', 'Variance', 'Histogram']


analysis = [
             inquirer.Checkbox('Analysis',
                               message="What kinds of ananlysis do you want to run on the layers chosen?",
                               choices= choicesList,
                               ),
             ]
respuesta = inquirer.prompt(analysis)


if 'Stats' in respuesta['Analysis']:
    print(getStats(df))

if 'Covariance' in respuesta['Analysis']:
    resultsDF = aggregateValues(df)
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            visualizeCovariance(df, norm = True)
            #    print(numpy.linalg.eigvals(resultsDF.replace('n/a', 0).astype(float)))
            resultsDF = pandas.DataFrame(preprocessing.MinMaxScaler().fit_transform(resultsDF), columns=resultsDF.columns, index=resultsDF.index)
#            print(norm.cov())
#            print(norm.describe())
            break
        elif visualizeInput == 'n':
            visualizeCovariance(df, norm = True)
            break



if 'Correlation' in respuesta['Analysis']:
    visualizeCorrelation(df)



if 'Variance' in respuesta['Analysis']:
    dataframe = aggregateValues(df)
    variance = dataframe.var()
    print(variance)
    mu = dataframe.mean().mean()
    print(mu)
    sigma = math.sqrt(variance)
    print(sigma)
    x = numpy.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.title("Normal Distribution "+names[answers['Layers'][0]], fontsize= 30)
    plt.plot(x,mlab.normpdf(x, mu, sigma))
    plt.show()

if 'Histogram' in respuesta['Analysis']:
    df = fileToDataframe(answers['Layers'][0])
    plotHistogram(df)

if 'Scatter Plot' in respuesta['Analysis']:
    dataframe = aggregateValues(df)
    plot(dataframe.sample(n=2000),names[answers['Layers'][0]],names[answers['Layers'][1]])
    # print(LA.eig(dataframe.as_matrix()))




## clustering and covariance matrices functions need to be inserted
###############################################################################
###############################################################################
###############################################################################
if 'Clustering' in respuesta['Analysis']:
    print('Clustering will be done with ', answers['Layers'])
if 'Distribution of Covariance Matrices' in respuesta['Analysis']:
    print('Distribution of covariance will be done with ', answers['Layers'])
