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
import plotly
import plotly.graph_objs as go

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

def make3dPlot(dataframe):
    z_data = dataframe
    colnames = (dataframe.columns.values)
    data = [
       go.Surface(
           z=z_data.as_matrix()
       )
    ]
    layout = go.Layout(
       title=colnames[2],
       autosize=False,
       width=500,
       height=500,
       margin=dict(
           l=65,
           r=50,
           b=65,
           t=90
       )
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig)

# clustering FUNCTIONS
def kmeans_decision_algo(df, cluster, normalized):
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

def plotHistogram(df, binNumbers):
    fig = plt.figure(figsize=(10,7))
    plt.xlabel('Value', fontsize= 20)
    plt.ylabel('Frequency', fontsize= 20)
    plt.title(df.columns.values.tolist()[-1], fontsize= 30)
    ax = fig.add_subplot(111)
    df = df.iloc[:,2].copy()
    ax = df.hist(figsize=(10,7), bins= binNumbers)
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


    # plt(x,y) -> plt(e2, e1) since first element should be on y axis and second element on x axis based on visualization paper
    plt.subplot(2,2,1)
    plt.title(r'$log(\sigma_1)\ vs\ \log(\sigma_2)$')
    plt.xlabel(r'$('+element2Value+')\ \log(\sigma_2)$')
    plt.ylabel(r'$('+element1Value+')\ \log(\sigma_1)$')
    plt.scatter(e2_log_arr, e1_log_arr)
    # plt.show()

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

    plt.subplot(2,2,2)
    plt.title(r'$log(\sigma_1)\ vs\ \rho_{12}$')
    plt.xlabel(r'$\rho_{12}$')
    plt.ylabel(r'$('+ element1Value +')\ \log(\sigma_1)$')
    plt.scatter(corr_d, e1_log_arr)

    # plt.show()

    # p23 correlation
    correlation2 = numpy.corrcoef(el2, el3)

    corr_d2=[]
    for x in range(0, len(correlation2)):
        for y in range(0, len(correlation2)):
            corr_d2.append(correlation2[x][y])

    #Graph to show element1 vs p23

    plt.subplot(2,2,3)
    plt.title(r'$log(\sigma_1)\ vs\ \rho_{23}$')
    plt.xlabel(r'$\rho_{23}$')
    plt.ylabel(r'$('+ element1Value +')\ \log(\sigma_1)$')
    plt.scatter(corr_d2, e1_log_arr)
    # plt.show()

    #Graph to show p12 vs p23

    plt.subplot(2,2,4)
    plt.title(r'$\rho_{12}\ vs\ \rho_{23}$')
    plt.xlabel(r'$\rho_{23}$')
    plt.ylabel(r'$\rho_{12}$')
    plt.scatter(corr_d2, corr_d)
    plt.tight_layout()
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
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Scatter Plot', 'Plot x vs y']
# chose multiple layers442
elif len(df) > 1:
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Log/Correlation Graph']
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


dataframe = aggregateValues(df)


if 'Stats' in respuesta['Analysis']:
    print(getStats(df))


if 'Covariance' in respuesta['Analysis']:
    aggregatedDataframe = aggregateValues(df)
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            visualizeCovariance(df, norm = True)
            break
        elif visualizeInput == 'n':
            visualizeCovariance(df, norm = False)
            break


if 'Correlation' in respuesta['Analysis']:
    visualizeCorrelation(df)


if 'Variance' in respuesta['Analysis']:
    variance = dataframe.var()
    # print(variance)
    mu = dataframe.mean().mean()
    # print(mu)
    sigma = math.sqrt(variance)
    # print(sigma)
    x = numpy.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.title("Normal Distribution "+names[answers['Layers'][0]], fontsize= 30)
    plt.plot(x,mlab.normpdf(x, mu, sigma))
    plt.show()


if 'Histogram' in respuesta['Analysis']:
    try:
        bins = int(input("How many bins for histogram (minimum 10)?"))
    except ValueError:
        print("Enter a number")
        bins = 0
    while(bins < 10):
        try:
            bins = int(input("How many bins for histogram?"))
        except ValueError:
            print("Enter a number")
            bins = 0
    plotHistogram(df[0], bins)


if 'Scatter Plot' in respuesta['Analysis']:
    # TODO - ask user for sample (integer)
    plot(dataframe.sample(n=2000),names[answers['Layers'][0]],names[answers['Layers'][1]])


if 'Log/Correlation Graph' in respuesta['Analysis']:
    length = int(input("How many rows/columns for matrix (single number)?"))        #for the log vs log graph
    logAndCorrelation(df[0], df[1], df[2], length)


# clustering
# TODO - crashes if 2 files are choosen
if 'Clustering' in respuesta['Analysis']:
    # normalize?
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            kmeans_decision_algo(dataframe, 3, True)
            break
        elif visualizeInput == 'n':
            kmeans_decision_algo(dataframe, 3, False)
            break

    wholeDf = pandas.merge(df[0], df[1], on=['x', 'y'])

    cAnswerLength = 0
    while cAnswerLength!=3:
        cluster = [
                     inquirer.Checkbox('Cluster',
                                       message="What 3 elements do you want to visualize?",
                                       choices= list(wholeDf.columns.values),
                                       ),
                     ]
        cAnswer = inquirer.prompt(cluster)
        cAnswerLength = len(cAnswer['Cluster'])


    # run function with choices choosen
    cluster_visualizer(wholeDf, cAnswer['Cluster'][0], cAnswer['Cluster'][1], cAnswer['Cluster'][2], 3)

# TODO - is the same as the scatterplot section
if 'Plot x vs y' in respuesta['Analysis']:
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            plot_x_y_normalize_all(dataframe, names[answers['Layers'][0]], names[answers['Layers'][1]], True, 0.5)
            break
        elif visualizeInput == 'n':
            plot_x_y_normalize_all(dataframe, names[answers['Layers'][0]], names[answers['Layers'][1]], False, 0.5)
            break

    # plot_x_y_normalize_individual(dataframe, names[answers['Layers'][0]], False, names[answers['Layers'][1]], False, 0.5)


if 'Plot layer' in respuesta['Analysis']:
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            plot_3_val_normalize_all(df[len(df)-1], 'x', 'y', names[answers['Layers'][len(df)-1]], True, 1)
            break
        elif visualizeInput == 'n':
            plot_3_val_normalize_all(df[len(df)-1], 'x', 'y', names[answers['Layers'][len(df)-1]], False, 1)
            break
    # plot_3_val_normalize_individual(df[len(df)-1], 'x', False, 'y', True, names[answers['Layers'][len(df)-1]], True, 1)


if '3d plot' in respuesta['Analysis']:
    # plot_three_val_3d(df[0], 'x', 'y', names[answers['Layers'][0]], 'blue' , 0.2)
    print(df[0].head())
    make3dPlot(df[0])


while True:
    visualizeTemp = input("Do you want to visualize the temp for the 24 hours?")
    if visualizeTemp == 'y':
        print("grab a cup of coffee (this is going to take a while!)")
        # hour 0
        hour00 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour00.xyz", header= None, delim_whitespace= True, encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp00']
        hour00.columns = names

        # hour 1
        hour01 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour01.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp01']
        hour01.columns = names
        completeDataframe = hour00.merge(hour01,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 2
        hour02 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour02.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp02']
        hour02.columns = names
        completeDataframe = completeDataframe.merge(hour02,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 3
        hour03 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour03.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp03']
        hour03.columns = names

        completeDataframe = completeDataframe.merge(hour03,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 4
        hour04 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour04.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp04']
        hour04.columns = names
        completeDataframe = completeDataframe.merge(hour04,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 5
        hour05 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour05.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp05']
        hour05.columns = names
        completeDataframe = completeDataframe.merge(hour05,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 6
        hour06 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour06.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp06']
        hour06.columns = names
        completeDataframe = completeDataframe.merge(hour06,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 7
        hour07 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour07.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp07']
        hour07.columns = names
        completeDataframe = completeDataframe.merge(hour07,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 8
        hour08 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour08.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp08']
        hour08.columns = names
        completeDataframe = completeDataframe.merge(hour08,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 9
        hour09 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour09.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp09']
        hour09.columns = names
        completeDataframe = completeDataframe.merge(hour09,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 10
        hour10 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour10.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp10']
        hour10.columns = names
        completeDataframe = completeDataframe.merge(hour10,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 11
        hour11 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour11.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp11']
        hour11.columns = names
        completeDataframe = completeDataframe.merge(hour11,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 12
        hour12 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour12.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp12']
        hour12.columns = names
        completeDataframe = completeDataframe.merge(hour12,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 13
        hour13 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour13.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp13']
        hour13.columns = names
        completeDataframe = completeDataframe.merge(hour13,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 14
        hour14 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour14.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp14']
        hour14.columns = names
        completeDataframe = completeDataframe.merge(hour14,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 15
        hour15 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour15.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp15']
        hour15.columns = names
        completeDataframe = completeDataframe.merge(hour15,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 16
        hour16 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour16.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp16']
        hour16.columns = names
        completeDataframe = completeDataframe.merge(hour16,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 17
        hour17 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour17.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp17']
        hour17.columns = names
        completeDataframe = completeDataframe.merge(hour17,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 18
        hour18 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour18.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp18']
        hour18.columns = names
        completeDataframe = completeDataframe.merge(hour18,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 19
        hour19 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour19.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp19']
        hour19.columns = names
        completeDataframe = completeDataframe.merge(hour19,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 20
        hour20 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour20.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp20']
        hour20.columns = names
        completeDataframe = completeDataframe.merge(hour20,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 21
        hour21 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour21.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp21']
        hour21.columns = names
        completeDataframe = completeDataframe.merge(hour21,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 22
        hour22 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour22.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp22']
        hour22.columns = names
        completeDataframe = completeDataframe.merge(hour22,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        # hour 23
        hour23 = pandas.DataFrame(pandas.read_csv("./temperature_xyz/temp_avg_hour23.xyz",header= None, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        names = ['Longitude', 'Latitude', 'Temp23']
        hour23.columns = names
        completeDataframe = completeDataframe.merge(hour23,left_on=["Longitude", "Latitude"],right_on=["Longitude", "Latitude"],how="outer")

        plot_hours_all(completeDataframe, 'Longitude', 'Latitude', completeDataframe.columns[2:26], color_array)
        break
    elif visualizeTemp == 'n':
        print("bless your soul!")
        break
