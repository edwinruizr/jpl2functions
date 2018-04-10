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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

color_array = ['#FF2052', '#E9D66B', '#00308F', '#3B3C36', '#85BB65', '#79C257',
               '#551B8C', '#B5651E', '#614051', '#669999', '#3B7A57',
              '#007FFF', '#8A0303', '#44D7A8', '#F0E130', '#007BA7',
              '#BFFF00', '#BFAFB2', '#FC8EAC', '#353839', '#FF77FF', '#C9FFE5',
              '#1CA9C9', '#DA8A67' ]

# dictionary holding filename and according element name
names = {'LP_GRS_Th_Global_2ppd.tif': 'Thorium',
    'LP_GRS_Fe_Global_2ppd.tif': 'Iron',
    'LP_GRS_H_Global_2ppd.tif' : 'Hydrogen',
    'resizedLOLA.xyz' : 'LOLA value'
    }


## FUNCTIONS
# TODO - figure how to implement this with the choices
def plot_hours_all(df, x, y, array_of_cols, colors):
    completeDataframe.head()
    fig = plt.figure(figsize = (20,15))
    ax = fig.add_subplot(111, projection='3d')
    for counter, i in enumerate(array_of_cols):
        p = ax.scatter(df[x], df[y], df[i], alpha = 0.25, c = colors[counter], label=i)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel('Temp')
    plt.legend()
    plt.title(x + ' vs ' + y)
    plt.show()

# clustering FUNCTIONS
def kmeans_decision_algo(df, cluster, normalized):
    print(df.head())
    if(normalized == True):
        kmean_algo_visualizer(normalize_df(df), cluster)
    else:
        kmean_algo_visualizer(df, cluster)

colors = ['red', 'blue', 'green', 'purple', 'cyan']

# should be called with a df of 3 columns
def kmean_algo_visualizer(df, cluster_size):
    kmeans = KMeans(n_clusters = cluster_size)
    kmeans.fit(df)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca(projection='3d')


    for l, c in zip(range(cluster_size), colors):
        current_members = (labels == l)
        current_center = centers[1]
        ax.scatter(df.iloc[current_members,0], df.iloc[current_members,1], df.iloc[current_members, 2], color=c, marker='.', alpha=0.025)

    ax.scatter(centers[:,0], centers[:,1], centers[:,2], marker='X', c='black', alpha=1)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    plt.show()



def col_number_getter(df, target):
    for i in range(len(df.columns)):
        if(df.columns[i] == target):
            return i

def cluster_visualizer(df, col_one, col_two, col_three, cluster_size):
    kmeans = KMeans(n_clusters = cluster_size)
    kmeans.fit(df)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca(projection='3d')
    col_one_number = col_number_getter(df, col_one)
    col_two_number = col_number_getter(df, col_two)
    col_three_number = col_number_getter(df, col_three)

    for l, c in zip(range(cluster_size), colors):
        current_members = (labels == l)
        current_center = centers[1]
        ax.scatter(df.iloc[current_members][col_one], df.iloc[current_members][col_two], df.iloc[current_members][col_three], color=c, marker='.', alpha=0.025)
    ax.scatter(centers[:,col_one_number], centers[:,col_two_number], centers[:,col_three_number], marker='X', c='black', alpha=1)
    ax.set_xlabel(col_one)
    ax.set_ylabel(col_two)
    ax.set_zlabel(col_three)
    plt.show()
# end of clustering FUNCTIONS

def series_convertor(x):
    if isinstance(x, pandas.Series):
        return x.to_frame()

def normalize_df(df):
    MMS = preprocessing.MinMaxScaler()
    normalized = MMS.fit_transform(df)
    normalized = pandas.DataFrame(normalized)
    normalized.columns = df.columns
    return normalized

def norm(df):
    normal = preprocessing.Normalizer()
    n = normal.fit_transform(df)
    n = pandas.DataFrame(n)
    n.columns = df.columns
    return n

def plot_x_y_normalize_all(df, x, y, normalizer, opacity):
    if(normalizer == True):
        df = normalize_df(df)
    plt.scatter(df[x], df[y], c=color_array[0], alpha = opacity)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(x + ' vs ' + y)
    plt.show()

def plot_x_y_normalize_individual(df, x, normalize_x, y, normalize_y, opacity):
    if(normalize_x == True):
        df['Normalized ' + x] = normalize_df(series_convertor(df[x])).values
        x = 'Normalized ' + x
    if(normalize_y == True):
        df['Normalized ' + y] = normalize_df(series_convertor(df[y])).values
        y = 'Normalized ' + y
    plt.scatter(df[x], df[y], c = color_array[0], alpha = opacity)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(x + ' vs ' + y)
    plt.show()


def plot_3_val_normalize_all(df, col_one, col_two, col_three, normalizer, opacity):
    if(normalizer == True):
        df = normalize_df(df)
    plt.scatter(df[col_one], df[col_two], c=df[col_three], alpha=opacity)
    plt.xlabel(col_one)
    plt.ylabel(col_two)
    plt.colorbar(label = col_three)
    plt.title(col_one + ' vs ' + col_two)
    plt.show()


def plot_3_val_normalize_individual(df, x, x_norm, y, y_norm, z, z_norm, opacity):
    if(x_norm == True):
        df['Normalized ' + x] = normalize_df(series_convertor(df[x])).values
        x = 'Normalized ' + x
    if(y_norm == True):
        df['Normalized ' + y] = normalize_df(series_convertor(df[y])).values
        y = 'Normalized ' + y
    if(z_norm == True):
        df['Normalized ' + z] = normalize_df(series_convertor(df[z])).values
        z = 'Normalized ' + z
    plt.scatter(df[x], df[y], c = df[z], alpha = opacity)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.colorbar(label = z)
    plt.title(x + ' vs ' + y)
    plt.show()

def plot_three_val_3d(df, x, y, z, color, alpha):
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df[x], df[y], df[z], alpha = alpha, marker = '.', c = color)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    if(type(color) != str):
        fig.colorbar(p)
    plt.show()

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


def logAndCorrelation(element1, element2, element3, length):
    element1Value = ""
    element2Value = ""
    element3Value = ""

    for key in element1.keys():
        if key != 'x' and key!='y':
            element1Value = key

    for key in element2.keys():
        if key != 'x' and key!='y':
            element2Value = key

    for key in element3.keys():
        if key!= 'x' and key!='y':
            element3Value = key

    xRange_e1 = element1['x'].max() - element1['x'].min()
    yRange_e1 = element1['y'].max() - element1['y'].min()

    section_e1 = math.ceil(xRange_e1/length)         # length specified by user
    ySection_e1 = math.ceil(yRange_e1/length)

    min_e1 = element1['x'].min()
    max_e1 = element1['x'].max()

    ymin_e1 = element1['y'].min()
    ymax_e1 = element1['y'].max()

    e1_d={}
    for x in range(0,length):
        for y in range(0,length):
            e1_d["matrix{0}{1}".format(x,y)]=element1[(min_e1+x*section_e1 <= element1['x']) & (element1['x'] < min_e1+(x+1)*section_e1) & (ymin_e1+y*ySection_e1 <= element1['y']) & (element1['y'] < ymin_e1+(y+1)*ySection_e1)]

    # Will contain the log(standard deviation) of each value in the matrix (Using this for scatterplot)
    e1_log_arr = []
    for key, value in e1_d.items():
        e1_key_std = numpy.std(e1_d[key][element1Value])
        e1_key_log = numpy.log(e1_key_std)
        e1_log_arr.append(e1_key_log)


    xRange_e2 = element2['x'].max() - element2['x'].min()
    yRange_e2 = element2['y'].max() - element2['y'].min()

    section_e2 = math.ceil(xRange_e2/length)
    ySection_e2 = math.ceil(yRange_e2/length)

    min_e2 = element2['x'].min()
    max_e2 = element2['x'].max()

    ymin_e2 = element2['y'].min()
    ymax_e2 = element2['y'].max()

    e2_d={}
    for x in range(0,length):
        for y in range(0,length):
            e2_d["matrix{0}{1}".format(x,y)]=element2[(min_e2+x*section_e2 <= element2['x']) & (element2['x'] < min_e2+(x+1)*section_e2) & (ymin_e2+y*ySection_e2 <= element2['y']) & (element2['y'] < ymin_e2+(y+1)*ySection_e2)]

    e2_log_arr = []
    for key, value in e2_d.items():
        e2_key_std = numpy.std(e2_d[key][element2Value])
        e2_key_log = numpy.log(e2_key_std)
        e2_log_arr.append(e2_key_log)

    # 3rd element
    xRange_e3 = element3['x'].max() - element3['x'].min()
    yRange_e3 = element3['y'].max() - element3['y'].min()

    section_e3 = math.ceil(xRange_e3/length)
    ySection_e3 = math.ceil(yRange_e3/length)

    min_e3 = element3['x'].min()
    max_e3 = element3['x'].max()

    ymin_e3 = element3['y'].min()
    ymax_e3 = element3['y'].max()

    e3_d={}
    for x in range(0,length):
        for y in range(0,length):
            e3_d["matrix{0}{1}".format(x,y)]=element3[(min_e3+x*section_e3 <= element3['x']) & (element3['x'] < min_e3+(x+1)*section_e3) & (ymin_e3+y*ySection_e3 <= element3['y']) & (element3['y'] < ymin_e3+(y+1)*ySection_e3)]

    e3_log_arr = []
    for key, value in e3_d.items():
        e3_key_std = numpy.std(e3_d[key][element3Value])
        e3_key_log = numpy.log(e3_key_std)
        e3_log_arr.append(e3_key_log)

    plt.title(r'$log(\sigma_1)\ vs\ \log(\sigma_2)$')
    plt.xlabel(r'$('+element2Value+')\ \log(\sigma_2)$')
    plt.ylabel(r'$('+element1Value+')\ \log(\sigma_1)$')
    # plt(x,y) -> plt(e2, e1) since first element should be on y axis and second element on x axis based on visualization paper
    plt.scatter(e2_log_arr, e1_log_arr)
    plt.show()

    ##### Correlation part #####
    el1 = element1.corr()
    el2 = element2.corr()
    el3 = element3.corr()

    first_el = el1[element1Value]
    second_el = el2[element2Value]
    third_el = el3[element3Value]

    # p12 correlation
    correlation = numpy.corrcoef(el1, el2)

    # This loops through the correlation matrix and puts all the points into a single array. (Correlation matrix is 6x6 and corr_d will have a size of 36)
    corr_d=[]
    for x in range(0, len(correlation)):
        for y in range(0, len(correlation)):
            corr_d.append(correlation[x][y])

    # This will change the size of e1_log_arr to have the same size as correlation (in order to plot on graph)
    d={}
    for x in range(0,6):
        for y in range(0,6):
            d["matrix{0}{1}".format(x,y)]=element1[(min_e1+x*section_e1 <= element1['x']) & (element1['x'] < min_e1+(x+1)*section_e1) & (ymin_e1+y*ySection_e1 <= element1['y']) & (element1['y'] < ymin_e1+(y+1)*ySection_e1)]


    # Will contain the log(standard deviation) of each value (Using this for correlation scatterplot)
    e1_log_arr = []
    for key, value in d.items():
        e1_key_std = numpy.std(d[key][element1Value])
        e1_key_log = numpy.log(e1_key_std)
        e1_log_arr.append(e1_key_log)

    # Graph to show element1 vs p12
    plt.title(r'$log(\sigma_1)\ vs\ \rho_{12}$')
    plt.xlabel(r'$\rho_{12}$')
    plt.ylabel(r'$('+ element1Value +')\ \log(\sigma_1)$')
    plt.scatter(corr_d, e1_log_arr)
    plt.show()

    # p23 correlation
    correlation2 = numpy.corrcoef(el2, el3)

    corr_d2=[]
    for x in range(0, len(correlation2)):
        for y in range(0, len(correlation2)):
            corr_d2.append(correlation2[x][y])

    #Graph to show element1 vs p23
    plt.title(r'$log(\sigma_1)\ vs\ \rho_{23}$')
    plt.xlabel(r'$\rho_{23}$')
    plt.ylabel(r'$('+ element1Value +')\ \log(\sigma_1)$')
    plt.scatter(corr_d2, e1_log_arr)
    plt.show()

    #Graph to show p12 vs p23
    plt.title(r'$\rho_{12}\ vs\ \rho_{23}$')
    plt.xlabel(r'$\rho_{23}$')
    plt.ylabel(r'$\rho_{12}$')
    plt.scatter(corr_d2, corr_d)
    plt.show()

## END OF FUNCTIONS


questions = [
             inquirer.Checkbox('Layers',
                               message="What layers do you want to analyze?",
                               choices=names.keys(),
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
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Distribution of Covariance Matrices', 'Scatter Plot', 'Plot x vs y']
# chose multiple layers442
elif len(df) > 1:
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Distribution of Covariance Matrices', 'Log/Correlation Graph']
# didn't choose any
elif len(df) == 0:
    print("You didn't choose any layers. Exiting.")
    exit(0)
# they chose 1 layer
else:
    choicesList = ['Stats', 'Variance', 'Histogram', 'Plot layer', '3d plot']


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

if 'Log/Correlation Graph' in respuesta['Analysis']:
    length = int(input("How many rows/columns for matrix (single number)?"))        #for the log vs log graph
    logAndCorrelation(df[0], df[1], df[2], length)

## clustering and covariance matrices functions need to be inserted
###############################################################################
###############################################################################
###############################################################################
if 'Clustering' in respuesta['Analysis']:
    wholeDf = pandas.merge(df[0], df[1], on=['x', 'y'])

    print(wholeDf.head())

    # dataframe = aggregateValues(df)
    # # TODO - ask for normalization
    # kmeans_decision_algo(dataframe, 3, True)

    # TODO - ask user what 3 parts of an element they want to visualize
    cluster_visualizer(wholeDf, 'x', 'y', 'Iron', 3)


if 'Plot x vs y' in respuesta['Analysis']:
    dataframe = aggregateValues(df)
    # TODO - ask for normalization
    plot_x_y_normalize_all(dataframe, names[answers['Layers'][0]], names[answers['Layers'][1]], False, 0.5)
    plot_x_y_normalize_individual(dataframe, names[answers['Layers'][0]], False, names[answers['Layers'][1]], False, 0.5)

if 'Plot layer' in respuesta['Analysis']:
    # TODO - ask for normalization
    plot_3_val_normalize_individual(df[len(df)-1], 'x', False, 'y', True, names[answers['Layers'][len(df)-1]], True, 1)
    # plot_3_val_normalize_all(df4, 'x', 'y', 'LOLA value', False, 1)

if '3d plot' in respuesta['Analysis']:
    plot_three_val_3d(df[0], 'x', 'y', names[answers['Layers'][0]], 'blue' , 0.2)

if 'Distribution of Covariance Matrices' in respuesta['Analysis']:
    print('Distribution of covariance will be done with ', answers['Layers'])
