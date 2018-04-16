import inquirer
import math
import os.path
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
from PIL import Image




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
def chris_needs_a_dataframe(df, elements, labels):
    return_frame = df[elements]
    return_frame['label'] = labels
    return return_frame

def normalize_choice(df, elements, norm_array):
    return_frame = pandas.DataFrame()
    for counter, i in enumerate(elements):
        if(norm_array[counter] == True):
            return_frame[i] = normalize_df(series_convertor(df[i]))
        else:
            return_frame[i] = df[i]
    return return_frame

def kmean(df, array_of_elements, array_of_normalization, cluster_size):
    kmeans = KMeans(n_clusters= cluster_size, random_state=1)
    filtered_df = normalize_choice(df[array_of_elements], array_of_elements, array_of_normalization)
    kmeans.fit(filtered_df)
    centers = kmeans.cluster_centers_
    return kmeans.labels_

def kmean_plot_2_val(df, xy_array, kmean_labels, cluster_size):
    fig = plt.figure(figsize=(24,15))
#         plt.scatter(df[x], df[y], color=c, marker='.', alpha=0.25, label=c)
    x = xy_array[0]
    y = xy_array[1]
    for l, c in zip(range(cluster_size), color_array):
        current_members = (kmean_labels == l)
        plt.scatter(df.iloc[current_members][x], df.iloc[current_members][y], color=c, marker='.', alpha=0.25, label=c)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


# TODO - figure how to implement this with the choices
def plot_hours_all(df, x, y, array_of_cols, colors):
    fileName = ''.join(array_of_cols)+''.join(colors)+x+y+'.png'
    if os.path.isfile(fileName):
        return fileName

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
    print(array_of_cols)
    plt.show()
    plt.savefig(fileName, bbox_inches='tight')
    return fileName

def make3dClusterPlot(df,colors_array):
    #for jupyter use
    # plotly.offline.init_notebook_mode()

    data = []

    for i in range(len(df['label'].unique())):
        name = df['label'].unique()[i]
        color = colors_array[i]
        x = df[ df['label'] == name ][list(df)[0]]
        y = df[ df['label'] == name ][list(df)[1]]
        z = df[ df['label'] == name ][list(df)[2]]

        trace = dict(
            name = name,
            x = x, y = y, z = z,
            type = "scatter3d",
            mode = 'markers',
            marker = dict( size=4, color=color, line=dict(width=0) )
        )
        data.append( trace )

        #cluster = dict(
        #    color = color,
        #    opacity = 0.0,
        #    type = "mesh3d",
        #    x = x, y = y, z = z )
        #data.append( cluster )

    layout = dict(
        title = '3d point clustering',
        legend=dict(
        x=0.75,
        y=1,
        traceorder='normal',
        font=dict(
            family='sans-serif',
            size=15,
            color='#000'
        ),
        bgcolor='#E2E2E2',
        bordercolor='#FFFFFF',
        borderwidth=2
        ),
        scene = dict(
        xaxis = dict( title=list(df_out)[0], zeroline=False ),
        yaxis = dict( title=list(df_out)[1], zeroline=False ),
        zaxis = dict( title=list(df_out)[2], zeroline=False ),
        ),
    )

    fig = dict(data=data, layout=layout)

    # plots and opens html page
    plotly.offline.plot(fig, filename='3d-scatter-cluster.html')

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
    # TODO - doesn't consider normalization (add extra bool value to function?)
    fileName = ''.join(list(df))+str(cluster_size)+'kmean.png'
    if os.path.isfile(fileName):
        return fileName
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
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName



def col_number_getter(df, target):
    for i in range(len(df.columns)):
        if(df.columns[i] == target):
            return i

def cluster_visualizer(df, col_one, col_two, col_three, cluster_size):
    # TODO - again should file name have normalization?
    fileName = ''.join(list(df)) + col_one + col_two + col_three + str(cluster_size) +'Cluster.png'
    if os.path.isfile(fileName):
        return fileName

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
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

# end of clustering FUNCTIONS

#check
def series_convertor(x):
    if isinstance(x, pandas.Series):
        return x.to_frame()

#check
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

#check
def plot_x_y_normalize_all(df, x, y, normalizer, opacity):
    if normalizer == True:
        normed = "Normalized"
    else:
        normed = ""

    fileName = ''.join(list(df)) + x + y + normed + str(opacity) +'scatterplot.png'
    if os.path.isfile(fileName):
        return fileName

    if(normalizer == True):
        df = normalize_df(df)
    plt.scatter(df[x], df[y], c=color_array[0], alpha = opacity)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(x + ' vs ' + y)
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

#check
def plot_x_y_normalize_individual(df, x, normalize_x, y, normalize_y, opacity):
    if normalize_x == True:
        normedX = "Normalized"
    else:
        normedX = ""

    if normalize_y == True:
        normedY = "Normalized"
    else:
        normedY = ""

    fileName = ''.join(list(df)) + x + normedX + y + normedY + str(opacity) +'scatterplot.png'
    if os.path.isfile(fileName):
        return fileName
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
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

#check
def plot_3_val_normalize_all(df, col_one, col_two, col_three, normalizer, opacity):
    if normalizer == True:
        normed = "AllNormalized"
    else:
        normed = ""
    fileName = ''.join(list(df)) + col_one + col_two + col_three + normed + str(opacity) +'Heatmap.png'
    if os.path.isfile(fileName):
        return fileName
    if(normalizer == True):
        df = normalize_df(df)
    plt.scatter(df[col_one], df[col_two], c=df[col_three], alpha=opacity)
    plt.xlabel(col_one)
    plt.ylabel(col_two)
    plt.colorbar(label = col_three)
    plt.title(col_one + ' vs ' + col_two)
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

#check
def plot_3_val_normalize_individual(df, x, x_norm, y, y_norm, z, z_norm, opacity):
    if x_norm == True:
        normedx = "Normalized"
    else:
        normedx = ""

    if y_norm == True:
        normedy = "Normalized"
    else:
        normedy = ""

    if z_norm == True:
        normedz = "Normalized"
    else:
        normedz = ""

    fileName = ''.join(list(df)) + normedx + x + normedy + y_norm + z_norm + z + str(opacity) +'Heatmap.png'
    if os.path.isfile(fileName):
        return fileName

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
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

#check
def plot_three_val_3d(df, x, y, z, color, alpha):
    fileName = ''.join(list(df)) + x + y + z + color + str(alpha) +'3Dscatterplot.png'
    if os.path.isfile(fileName):
        return fileName

    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df[x], df[y], df[z], alpha = alpha, marker = '.', c = color)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    if(type(color) != str):
        fig.colorbar(p)
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

'''
I started adding here

'''
def plot_3_val_3d_normalize_all(df, x, y, z, normalize, color, alpha):
    if(normalize == True):
        df = normalize_df(df)
    fig = plt.figure(figsize = (20,15))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df[x], df[y], df[z], alpha = alpha, marker = '.', c = color)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    if(type(color) != str):
        fig.colorbar(p)
    plt.show()

def plot_3_val_3d_normalize_individual(df, x, x_norm, y, y_norm, z, z_norm, color, opacity):
    if(x_norm == True):
        df['Normalized ' + x] = normalize_df(series_convertor(df[x])).values
        x = 'Normalized ' + x
    if(y_norm == True):
        df['Normalized ' + y] = normalize_df(series_convertor(df[y])).values
        y = 'Normalized ' + y
    if(z_norm == True):
        df['Normalized ' + z] = normalize_df(series_convertor(df[z])).values
        z = 'Normalized ' + z
    fig = plt.figure(figsize = (20, 15))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df[x], df[y], df[z], alpha = opacity, marker = '.', c = color)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()

def kmean_plot(df, xyz_array, kmean_labels, cluster_size):
    fig = plt.figure(figsize=(24,15))
    ax = fig.gca(projection='3d')
    x = xyz_array[0]
    y = xyz_array[1]
    z = xyz_array[2]
    for l, c in zip(range(cluster_size), color_array):
        current_members = (kmean_labels == l)
        ax.scatter(df.iloc[current_members][x], df.iloc[current_members][y], df.iloc[current_members][z], color=c, marker='.', alpha=0.25, label=c)
    ax.set_xlabel(x)
    ax.set_xlabel(y)
    ax.set_xlabel(z)
    ax.legend()
    ax.set_title('Clustering displayed in ' + x + ' ' + y + ' ' + z + ' space')
    
'''
Ended here
'''

def plotHistogram(df, binNumbers):
    fileName = df.columns.values.tolist()[-1] + 'HistogramWith' + str(binNumbers) + 'Bins.png'
    if os.path.isfile(fileName):
        return fileName

    fig = plt.figure(figsize=(10,7))
    plt.xlabel('Value', fontsize= 20)
    plt.ylabel('Frequency', fontsize= 20)
    plt.title(df.columns.values.tolist()[-1], fontsize= 30)
    ax = fig.add_subplot(111)
    df = df.iloc[:,2].copy()
    ax = df.hist(figsize=(10,7), bins= binNumbers)
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName


def visualizeVariance(dataframe, elementName):
    fileName = ''.join(list(dataframe)) + elementName +'Variance.png'
    if os.path.isfile(fileName):
        return fileName

    variance = dataframe.var()
    mu = dataframe.mean().mean()
    sigma = math.sqrt(variance)
    x = numpy.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.title("Normal Distribution "+ elementName, fontsize= 30)
    plt.plot(x,mlab.normpdf(x, mu, sigma))
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

def plot(df, xname, yname):
    fileName = ''.join(list(df)) + xname + yname + 'EqualAspectScatterplot.png'
    if os.path.isfile(fileName):
        return fileName
    ax = df.plot(kind='scatter', x=xname, y=yname)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.suptitle('scatter plot', fontsize= 30)
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

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



# input list of dataframes -> output dataframe which is a correlation matrix of dataframes passed (if you just want text)
def getCorrelation(listOfDataframes):
    valuesDf = aggregateValues(listOfDataframes)
    return valuesDf.corr()

# if you want correlation visualization
def visualizeCorrelation(listOfDataframes):
    valuesDf = aggregateValues(listOfDataframes)

    fileName = ''.join(list(valuesDf)) + 'CorrelationMatrix.png'
    if os.path.isfile(fileName):
        return fileName

    correlations = valuesDf.corr()
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
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

def getMin(df):
    min = df.min().min()
    return min


def getMax(df):
    max = df.max().max()
    return max

def visualizeCovariance(listOfDataframes, norm = False):
    if norm == True:
        normed = "Normalized"
    else:
        normed = ""

    valuesDf = aggregateValues(listOfDataframes)
    fileName = ''.join(list(valuesDf)) + normed +'CovarianceMatrix.png'
    if os.path.isfile(fileName):
        return fileName

    if norm:
        valuesDF = pandas.DataFrame(preprocessing.MinMaxScaler().fit_transform(valuesDf), columns=valuesDf.columns, index=valuesDf.index)

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
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

def logAndCorrelation(element1, element2, element3, row, column):


    element1Value = ""
    element2Value = ""
    element3Value = ""

    for key in element1.keys():
        if key != 'Lat' and key!='Long':
            element1Value = key

    for key in element2.keys():
        if key != 'Lat' and key!='Long':
            element2Value = key

    for key in element3.keys():
        if key!= 'Lat' and key!='Long':
            element3Value = key

    fileName = element1Value + element2Value + element3Value + str(row*column) +'SubsectionsLogAndCorrelationPlot.png'
    if os.path.isfile(fileName):
        return fileName


    xRange_e1 = element1['Lat'].max() - element1['Lat'].min()
    yRange_e1 = element1['Long'].max() - element1['Long'].min()

    section_e1 = math.ceil(xRange_e1/row)         # length specified by user
    ySection_e1 = math.ceil(yRange_e1/column)

    min_e1 = element1['Lat'].min()
    max_e1 = element1['Lat'].max()

    ymin_e1 = element1['Long'].min()
    ymax_e1 = element1['Long'].max()

    e1_d={}
    for x in range(0,row):
        for y in range(0,column):
            e1_d["matrix{0}{1}".format(x,y)]=element1[(min_e1+x*section_e1 <= element1['Lat']) & (element1['Lat'] < min_e1+(x+1)*section_e1) & (ymin_e1+y*ySection_e1 <= element1['Long']) & (element1['Long'] < ymin_e1+(y+1)*ySection_e1)]

    # Will contain the log(standard deviation) of each value in the matrix (Using this for scatterplot)
    e1_log_arr = []
    for key, value in e1_d.items():
        e1_key_std = numpy.std(e1_d[key][element1Value])
        e1_key_log = numpy.log(e1_key_std)
        e1_log_arr.append(e1_key_log)


    xRange_e2 = element2['Lat'].max() - element2['Lat'].min()
    yRange_e2 = element2['Long'].max() - element2['Long'].min()

    section_e2 = math.ceil(xRange_e2/row)
    ySection_e2 = math.ceil(yRange_e2/column)

    min_e2 = element2['Lat'].min()
    max_e2 = element2['Lat'].max()

    ymin_e2 = element2['Long'].min()
    ymax_e2 = element2['Long'].max()

    e2_d={}
    for x in range(0,row):
        for y in range(0,column):
            e2_d["matrix{0}{1}".format(x,y)]=element2[(min_e2+x*section_e2 <= element2['Lat']) & (element2['Lat'] < min_e2+(x+1)*section_e2) & (ymin_e2+y*ySection_e2 <= element2['Long']) & (element2['Long'] < ymin_e2+(y+1)*ySection_e2)]

    e2_log_arr = []
    for key, value in e2_d.items():
        e2_key_std = numpy.std(e2_d[key][element2Value])
        e2_key_log = numpy.log(e2_key_std)
        e2_log_arr.append(e2_key_log)

    # 3rd element
    xRange_e3 = element3['Lat'].max() - element3['Lat'].min()
    yRange_e3 = element3['Long'].max() - element3['Long'].min()

    section_e3 = math.ceil(xRange_e3/row)
    ySection_e3 = math.ceil(yRange_e3/column)

    min_e3 = element3['Lat'].min()
    max_e3 = element3['Lat'].max()

    ymin_e3 = element3['Long'].min()
    ymax_e3 = element3['Long'].max()

    e3_d={}
    for x in range(0,row):
        for y in range(0,column):
            e3_d["matrix{0}{1}".format(x,y)]=element3[(min_e3+x*section_e3 <= element3['Lat']) & (element3['Lat'] < min_e3+(x+1)*section_e3) & (ymin_e3+y*ySection_e3 <= element3['Long']) & (element3['Long'] < ymin_e3+(y+1)*ySection_e3)]

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
            d["matrix{0}{1}".format(x,y)]=element1[(min_e1+x*section_e1 <= element1['Lat']) & (element1['Lat'] < min_e1+(x+1)*section_e1) & (ymin_e1+y*ySection_e1 <= element1['Long']) & (element1['Long'] < ymin_e1+(y+1)*ySection_e1)]


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


    #Graph to show p12 vs p23

    plt.subplot(2,2,4)
    plt.title(r'$\rho_{12}\ vs\ \rho_{23}$')
    plt.xlabel(r'$\rho_{23}$')
    plt.ylabel(r'$\rho_{12}$')
    plt.scatter(corr_d2, corr_d)
    plt.tight_layout()
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName

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
        tempDf = fileToDataframe(answer)
        # TODO - call conversion function HERE
        tempDf = tempDf.rename(columns={"x": "Lat", "y": "Long"})
        df.append(tempDf)
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
            Image.open(visualizeCovariance(df, norm = True)).show()
            break
        elif visualizeInput == 'n':
            Image.open(visualizeCovariance(df, norm = False)).show()
            break


if 'Correlation' in respuesta['Analysis']:
    # normalization doesn't matter (produces the same output)
    img = Image.open(visualizeCorrelation(df))
    img.show()


if 'Variance' in respuesta['Analysis']:
    img = Image.open(visualizeVariance(dataframe, names[answers['Layers'][0]]))
    img.show()


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
    fileName = plotHistogram(df[0], bins)
    Image.open(fileName).show()

if 'Scatter Plot' in respuesta['Analysis']:
    # TODO - ask user for sample (integer)
    fileName = plot(dataframe.sample(n=2000),names[answers['Layers'][0]],names[answers['Layers'][1]])
    Image.open(fileName).show()

if 'Log/Correlation Graph' in respuesta['Analysis']:
    # To show up the first time
    row = int(input("How many rows for matrix? "))
    column = int(input("How many columns for matrix? "))
    logAndCorrelation(df[0], df[1], df[2], row, column)

    # To allow user to change the size of the length without having to reload the program each time (Dr. Zhu wanted to implement this)
    # Also to allow user to exit if they want to stop looking at graphs
    while True:
        userChoice = input("\nWould you like to continue to view graphs? (y/n)")
        if userChoice == 'y':
            row = int(input("How many rows for matrix? "))
            column = int(input("How many columns for matrix? "))
            fileName = logAndCorrelation(df[0], df[1], df[2], row, column)
            Image.open(fileName).show()
        elif userChoice == 'n':
            exit(0)

# clustering
# TODO - crashes if 2 files are choosen
if 'Clustering' in respuesta['Analysis']:
    # ask for number of clusters
    while True:
        try:
            clusterSize = int(input("How many clusters? (2-10)"))
        except ValueError:
            print("Enter a number!")
            clusterSize = 0
        if 2 <= clusterSize <= 10:
            break

    # normalize?
    normalizeBoolList = []
    for i in range(len(answers['Layers'])):

        while True:
            visualizeInput = input("Do you want to normalize "+ names[answers['Layers'][i]] +" data? (y/n)")
            if visualizeInput == 'y':
                normalizeBoolList.append(True)
                break
            elif visualizeInput == 'n':
                normalizeBoolList.append(False)
                break

    # print(normalizeBoolList)

    # TODO - function parameters will be changed
    # while True:
    #     visualizeInput = input("Do you want to normalize the data? (y/n)")
    #     if visualizeInput == 'y':
    #         fileName = kmeans_decision_algo(dataframe, clusterSize, True)
    #         Image.open(fileName).show()
    #         break
    #     elif visualizeInput == 'n':
    #         fileName = kmeans_decision_algo(dataframe, clusterSize, False)
    #         Image.open(fileName).show()
    #         break
    wholeDf = df[0]
    for x in range(0, len(df)-1):
        wholeDf = pandas.merge(wholeDf, df[x+1], on=['Lat', 'Long'])


    cAnswerLength = 0
    while cAnswerLength!=3 and cAnswerLength!=2:
        cluster = [
                     inquirer.Checkbox('Cluster',
                                       message="What 3 elements do you want to visualize?",
                                       choices= list(wholeDf.columns.values),
                                       ),
                     ]
        cAnswer = inquirer.prompt(cluster)
        cAnswerLength = len(cAnswer['Cluster'])

    elementsList = []
    for layer in answers['Layers']:
        elementsList.append(names[layer])
    labels = kmean(wholeDf, elementsList, normalizeBoolList, clusterSize)
    # run function with choices choosen
    if cAnswerLength == 2:
        kmean_plot_2_val(wholeDf, cAnswer['Cluster'], labels, clusterSize)

    if cAnswerLength == 3:
        make3dClusterPlot(chris_needs_a_dataframe(wholeDf,cAnswer['Cluster'],labels),color_array)
    # fileName =   cluster_visualizer(wholeDf, cAnswer['Cluster'][0], cAnswer['Cluster'][1], cAnswer['Cluster'][2], 3)
    # Image.open(fileName).show()

# TODO - is the same as the scatterplot section
if 'Plot x vs y' in respuesta['Analysis']:
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            fileName = plot_x_y_normalize_all(dataframe, names[answers['Layers'][0]], names[answers['Layers'][1]], True, 0.5)
            Image.open(fileName).show()
            break
        elif visualizeInput == 'n':
            fileName = plot_x_y_normalize_all(dataframe, names[answers['Layers'][0]], names[answers['Layers'][1]], False, 0.5)
            Image.open(fileName).show()
            break

    # plot_x_y_normalize_individual(dataframe, names[answers['Layers'][0]], False, names[answers['Layers'][1]], False, 0.5)


if 'Plot layer' in respuesta['Analysis']:
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            fileName = plot_3_val_normalize_all(df[len(df)-1], 'Lat', 'Long', names[answers['Layers'][len(df)-1]], True, 1)
            Image.open(fileName).show()
            break
        elif visualizeInput == 'n':
            fileName = plot_3_val_normalize_all(df[len(df)-1], 'Lat', 'Long', names[answers['Layers'][len(df)-1]], False, 1)
            Image.open(fileName).show()
            break
    # plot_3_val_normalize_individual(df[len(df)-1], 'x', False, 'y', True, names[answers['Layers'][len(df)-1]], True, 1)





if '3d plot' in respuesta['Analysis']:
    fileName = plot_three_val_3d(df[0], 'Lat', 'Long', names[answers['Layers'][0]], 'blue' , 0.2)
    Image.open(fileName).show()
    print(df[0].head())
    # TODO - make3dPlot crashes
    make3dPlot(df[0], color_array)


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

        temp_cols = ['Temp00', 'Temp01', 'Temp02', 'Temp03', 'Temp04', 'Temp05', 'Temp06','Temp07', 'Temp08', 'Temp09', 'Temp10', 'Temp11', 'Temp12', 'Temp13','Temp14', 'Temp15', 'Temp16', 'Temp17', 'Temp18', 'Temp19', 'Temp20', 'Temp21', 'Temp22', 'Temp23']

        for col in temp_cols:
            completeDataframe[col] = completeDataframe[col].interpolate(method='linear')
        
        fileName = plot_hours_all(completeDataframe, 'Longitude', 'Latitude', completeDataframe.columns[2:26], color_array)
        Image.open(fileName).show()
        break
    elif visualizeTemp == 'n':
        print("bless your soul!")
        break
