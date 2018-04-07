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


def logVsLog(element1, element2):
    element1Value = ""
    element2Value = ""
    for key in element1.keys():
        if key != 'x' and key!='y':
            element1Value = key

    for key in element2.keys():
        if key != 'x' and key!='y':
            element2Value = key

    xRange_e1 = element1['x'].max() - element1['x'].min()
    yRange_e1 = element1['y'].max() - element1['y'].min()

    section_e1 = math.ceil(xRange_e1/10)         #Manually put 10 to have 10 x 10 matrix
    ySection_e1 = math.ceil(yRange_e1/10)

    min_e1 = element1['x'].min()
    max_e1 = element1['x'].max()

    ymin_e1 = element1['y'].min()
    ymax_e1 = element1['y'].max()

    d={}
    for x in range(0,10):
        for y in range(0,10):
            d["matrix{0}{1}".format(x,y)]=element1[(min_e1+x*section_e1 <= element1['x']) & (element1['x'] < min_e1+(x+1)*section_e1) & (ymin_e1+y*ySection_e1 <= element1['y']) & (element1['y'] < ymin_e1+(y+1)*ySection_e1)]

    # Will contain the log(standard deviation) of each value in the 10x10 matrix (Using this for scatterplot)
    fe_log_arr = []
    for key, value in d.items():
        fe_key_std = numpy.std(d[key][element1Value])
        fe_key_log = numpy.log(fe_key_std)
        fe_log_arr.append(fe_key_log)


    xRange_e2 = element2['x'].max() - element2['x'].min()
    yRange_e2 = element2['y'].max() - element2['y'].min()

    section_e2 = math.ceil(xRange_e2/10)
    ySection_e2 = math.ceil(yRange_e2/10)

    min_e2 = element2['x'].min()
    max_e2 = element2['x'].max()

    ymin_e2 = element2['y'].min()
    ymax_e2 = element2['y'].max()

    h_d={}
    for x in range(0,10):
        for y in range(0,10):
            h_d["matrix{0}{1}".format(x,y)]=element2[(min_e2+x*section_e2 <= element2['x']) & (element2['x'] < min_e2+(x+1)*section_e2) & (ymin_e2+y*ySection_e2 <= element2['y']) & (element2['y'] < ymin_e2+(y+1)*ySection_e2)]

    h_log_arr = []
    for key, value in h_d.items():
        h_key_std = numpy.std(h_d[key][element2Value])
        h_key_log = numpy.log(h_key_std)
        h_log_arr.append(h_key_log)

    plt.title(r'$log(\sigma_1)\ vs\ \log(\sigma_2)$')
    plt.xlabel(r'$('+element1Value+')\ \log(\sigma_2)$')
    plt.ylabel(r'$('+element2Value+')\ \log(\sigma_1)$')
    # plt(x,y) -> plt(hydrogen, iron) since first element should be on y axis and second element on x axis based on visualization paper
    plt.scatter(h_log_arr, fe_log_arr)
    plt.show()

    fe = element1.corr()
    h = element2.corr()

    iron = fe[element1Value]
    hydrogen = h[element2Value]

    correlation = numpy.corrcoef(fe, h)

    # This loops through the correlation matrix and puts all the points into a single array.
    corr_d=[]
    for x in range(0, len(correlation)):
        for y in range(0, len(correlation)):
            corr_d.append(correlation[x][y])

    d={}
    for x in range(0,6):
        for y in range(0,6):
            d["matrix{0}{1}".format(x,y)]=element1[(min_e1+x*section_e1 <= element1['x']) & (element1['x'] < min_e1+(x+1)*section_e1) & (ymin_e1+y*ySection_e1 <= element1['y']) & (element1['y'] < ymin_e1+(y+1)*ySection_e1)]

    # Will contain the log(standard deviation) of each value in the 10x10 matrix (Using this for scatterplot)
    fe_log_arr = []
    for key, value in d.items():
        fe_key_std = numpy.std(d[key][element1Value])
        fe_key_log = numpy.log(fe_key_std)
        fe_log_arr.append(fe_key_log)

    plt.title(r'$log(\sigma_1)\ vs\ \rho_{12}$')
    plt.xlabel(r'$\rho_{12}$')
    plt.ylabel(r'$('+ element1Value +')\ \log(\sigma_1)$')
    plt.scatter(corr_d, fe_log_arr)
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
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Distribution of Covariance Matrices', 'Scatter Plot', 'Log Graph']
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

if 'Log Graph' in respuesta['Analysis']:
    logVsLog(df[0], df[1])


## clustering and covariance matrices functions need to be inserted
###############################################################################
###############################################################################
###############################################################################
if 'Clustering' in respuesta['Analysis']:
    print('Clustering will be done with ', answers['Layers'])
if 'Distribution of Covariance Matrices' in respuesta['Analysis']:
    print('Distribution of covariance will be done with ', answers['Layers'])
