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
names = {
    'Thorium' : 'thorium.xyz',
    'Iron' : 'iron.xyz',
    'Hydrogen' : 'hydrogen.xyz',
    'LOLA (elevation)' : 'resizedLOLA.xyz',
    'Temp Hour 00' : 'temp_avg_hour00.xyz',
    'Temp Hour 01' : 'temp_avg_hour01.xyz',
    'Temp Hour 02' : 'temp_avg_hour02.xyz',
    'Temp Hour 03' : 'temp_avg_hour03.xyz',
    'Temp Hour 04' : 'temp_avg_hour04.xyz',
    'Temp Hour 05' : 'temp_avg_hour05.xyz',
    'Temp Hour 06' : 'temp_avg_hour06.xyz',
    'Temp Hour 07' : 'temp_avg_hour07.xyz',
    'Temp Hour 08' : 'temp_avg_hour08.xyz',
    'Temp Hour 09' : 'temp_avg_hour09.xyz',
    'Temp Hour 10' : 'temp_avg_hour10.xyz',
    'Temp Hour 11' : 'temp_avg_hour11.xyz',
    'Temp Hour 12' : 'temp_avg_hour12.xyz',
    'Temp Hour 13' : 'temp_avg_hour13.xyz',
    'Temp Hour 14' : 'temp_avg_hour14.xyz',
    'Temp Hour 15' : 'temp_avg_hour15.xyz',
    'Temp Hour 16' : 'temp_avg_hour16.xyz',
    'Temp Hour 17' : 'temp_avg_hour17.xyz',
    'Temp Hour 18' : 'temp_avg_hour18.xyz',
    'Temp Hour 19' : 'temp_avg_hour19.xyz',
    'Temp Hour 20' : 'temp_avg_hour20.xyz',
    'Temp Hour 21' : 'temp_avg_hour21.xyz',
    'Temp Hour 22' : 'temp_avg_hour22.xyz',
    'Temp Hour 23' : 'temp_avg_hour23.xyz'
    }



# input file (tif or xyz) -> output pandas dataframe
def fileToDataframe(file, columnName):
    if '.xyz' in file:
        df = pandas.DataFrame(pandas.read_csv(file, delim_whitespace= True,encoding="utf-8-sig", dtype=numpy.float64))
        df = df.interpolate()
    else:
        df = gr.from_file(file).to_pandas()
        df = df[["x", "y", "value"]].copy()
    df.columns =['Lat', 'Long', columnName]
    return df


def aggregateValues(listOfDataframes):
    """
        input list of dataframe -> outputs single dataframe (multiple dataframes -> single dataframe)
        concatenation of values of elements not merging based on lat long values
    """
    for index, df in enumerate(listOfDataframes):
        if index == 0:
            valuesDf = df.iloc[:,2].copy()
            valuesDf = valuesDf.to_frame()
        else:
            valuesDf = pandas.concat([valuesDf, df.iloc[:,2].copy()], axis=1)
    return valuesDf

def combineOnLatLong(listOfDataframes):
    """
        input list of dataframe -> outputs single dataframe (multiple dataframes -> single dataframe)
        merges based on lat long values
    """
    wholeDf = df[0]
    for x in range(0, len(df)-1):
        wholeDf = pandas.merge(wholeDf, df[x+1], how='inner', on=['Lat', 'Long'])

    for answer in answers['Layers']:
        wholeDf=wholeDf[wholeDf[names[answer]].notnull()]
    return wholeDf


def getStats(dataframe):
    """
        input dataframe -> outputs dataframe of mean, std, min, 25%, 50%, 75%, and max
    """
    print(type(dataframe.describe()))
    return dataframe.describe()


def getMin(df):
    min = df.min().min()
    return min


def getMax(df):
    max = df.max().max()
    return max


def visualizeCovariance(aggregatedDataframe, elementNamesList, norm = False):

    if norm == True:
        normed = "Normalized"
    else:
        normed = ""


    fileName = ''.join(elementNamesList) + normed +'CovarianceMatrix.png'
    if os.path.isfile(fileName):
        return fileName

    if norm:
        aggregatedDataframe = pandas.DataFrame(preprocessing.MinMaxScaler().fit_transform(aggregatedDataframe), columns=aggregatedDataframe.columns, index=aggregatedDataframe.index)

    covariance = aggregatedDataframe.cov()
    max = getMax(covariance)
    min = getMin(covariance)
    fig = plt.figure(figsize=(10,7))
    plt.title("Covariance Matrix", fontsize= 30)
    ax = fig.add_subplot(111)
    cax = ax.matshow(covariance, cmap=cm.gist_earth, alpha=0.7, vmin=getMin(covariance), vmax=getMax(covariance))
    for (i, j), z in numpy.ndenumerate(covariance):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center')
    fig.colorbar(cax)
    ticks = numpy.arange(0,len(elementNamesList),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(covariance.columns)
    ax.set_yticklabels(covariance.columns)
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName


def visualizeCorrelation(aggregatedDataframe, elementNamesList):
    fileName = ''.join(list(elementNamesList)) + 'CorrelationMatrix.png'
    if os.path.isfile(fileName):
        return fileName

    correlations = aggregatedDataframe.corr()
    fig = plt.figure(figsize=(10,7))
    plt.title("Correlation Matrix", fontsize= 30)
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, cmap=cm.gist_earth,alpha=0.7,vmin=getMin(correlations), vmax=getMax(correlations))
    for (i, j), z in numpy.ndenumerate(correlations):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center')
    fig.colorbar(cax)
    ticks = numpy.arange(0,len(elementNamesList),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(correlations.columns)
    ax.set_yticklabels(correlations.columns)
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


def plotHistogram(df, binNumbers):
    fileName = df.columns.values.tolist()[-1] + 'HistogramWith' + str(binNumbers) + 'Bins.png'
    if os.path.isfile(fileName):
        return fileName

    fig = plt.figure(figsize=(10,7))
    plt.xlabel('Value', fontsize= 20)
    plt.ylabel('Frequency', fontsize= 20)
    plt.title(df.columns.values.tolist()[-1], fontsize= 30)
    ax = fig.add_subplot(111)
    df = df.iloc[:,0].copy()
    ax = df.hist(figsize=(10,7), bins= binNumbers)
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName


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


def make3dPlot(df,xcol,ycol,zcol,color,alpha):
    data = []
    trace = dict(
        name = 'point',
        x = df[xcol], y = df[ycol], z = df[zcol],
        type = "scatter3d",
        mode = 'markers',
        marker = dict( opacity=alpha, size=4, color=color, line=dict(width=0) )
    )
    data.append( trace )

    layout = dict(
        title = 'Plot of ' + xcol + '(x) ' + ycol + '(y) ' + zcol + '(z)',
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
        xaxis = dict( title=xcol, zeroline=False ),
        yaxis = dict( title=ycol, zeroline=False ),
        zaxis = dict( title=zcol, zeroline=False ),
        ),
    )

    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename='3d-scatter-plot.html')


def plot_x_y_normalize_all(df, x, y, normalizer, equalAspect,opacity):
    if normalizer == True:
        normed = "Normalized"
    else:
        normed = ""

    if equalAspect == True:
        equal = "EqualAspect"
    else:
        equal = ""

    fileName = ''.join(list(df)) + x + y + normed + equal + str(opacity) +'scatterplot.png'
    if os.path.isfile(fileName):
        return fileName

    if(normalizer == True):
        df = normalize_df(df)
    plt.scatter(df[x], df[y], c=color_array[0], alpha = opacity)
    if equalAspect == True:
        plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(x + ' vs ' + y, fontsize= 25)
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.show()
    return fileName



def dataframe_label_assign(df, elements, labels):
    return_frame = df[elements]
    return_frame['label'] = labels
    return return_frame



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


    layout = dict(
        title = 'Cluster Plot of ' + list(df)[0] + '(x) ' + list(df)[1] + '(y) ' + list(df)[2] + '(z)',
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
        xaxis = dict( title=list(df)[0], zeroline=False ),
        yaxis = dict( title=list(df)[1], zeroline=False ),
        zaxis = dict( title=list(df)[2], zeroline=False ),
        ),
    )

    fig = dict(data=data, layout=layout)

    # plots and opens html page
    plotly.offline.plot(fig, filename='3d-scatter-cluster.html')


def series_converter(x):
    if isinstance(x, pandas.Series):
        return x.to_frame()



def normalize_df(df):
    MMS = preprocessing.MinMaxScaler()
    normalized = MMS.fit_transform(df)
    normalized = pandas.DataFrame(normalized)
    normalized.columns = df.columns
    return normalized

def normalize_choice(df, elements, norm_array):
    return_frame = pandas.DataFrame()
    for counter, i in enumerate(elements):
        if(norm_array[counter] == True):
            return_frame[i] = normalize_df(series_converter(df[i]))
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

# subsamples dataframe by percent, ie subsamplePercent = 5 means 5% of values in dataframe
def dataframeSubsampler(df,subsamplePercent):
    subsampleVal = 100/subsamplePercent
    df_out = df[0::int(subsampleVal)]
    return df_out


def plotAllTemp(df,xcol,ycol,array_of_cols,color_array,):
    #for jupyter use
    #plotly.offline.init_notebook_mode()

    data = []

    #check if dataframe is too large for plotting
    subsampleVal = 100
    if((len(df)*len(array_of_cols)) > 500000):
        subsampleVal = int((500000*100)/(len(df)*len(array_of_cols)))
        df = dataframeSubsampler(df,subsampleVal)

    for counter, i in enumerate(array_of_cols):

        trace = dict(
            name = i,
            x = df[xcol], y = df[ycol], z = df[i],
            type = "scatter3d",
            mode = 'markers',
            marker = dict( opacity=0.9, size=4, color=color_array[counter], line=dict(width=0) )
        )
        data.append( trace )


    layout = dict(
        title = 'Plot of all Temperatures(' + str(subsampleVal) + '% of data subsampled)',
        annotations=[
        dict(
            x=0.79,
            y=1.02,
            align="right",
            valign="top",
            text='Temp Layer',
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="middle",
            yanchor="top"
        )
        ],
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
        xaxis = dict( title=xcol, zeroline=False ),
        yaxis = dict( title=ycol, zeroline=False ),
        zaxis = dict( title='Z value', zeroline=False ),
        ),
    )

    fig = dict(data=data, layout=layout)

    # plots and opens html page
    plotly.offline.plot(fig, filename='3d-temp-plot.html')


###ONE LAYER VISUAL###
def logAndCorrelation1(element1, row, column, bins):
    element1Value = ""

    for key in element1.keys():
        if key != 'Lat' and key!='Long':
            element1Value = key
    fileName = element1Value + str(row*column) +'SubsectionsLogAndCorrelationPlot.png'
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

        #variance
    covMatrices1el = []
    for key, value in e1_d.items():
        covariance1 = numpy.cov(e1_d[key][element1Value])
        covMatrices1el.append(covariance1)
        #print(len(covMatrices1el))


    #Histogram
    plt.figure(1)
    plt.title(r'$Frequency\ vs\ log(\sigma_1)$')
    plt.xlabel(r'$log(\sigma_1)$')
    plt.ylabel(r'Frequency')
    plt.hist(e1_log_arr, bins, edgecolor='black', linewidth=1.2)
    plt.tight_layout()
    plt.show()

    #Variance
    plt.figure(2)
    plt.title(r'$Frequency\ vs\ Effective\ Variance$')
    plt.xlabel(r'$Effective Variance$')
    plt.ylabel(r'Frequency')
    plt.hist(covMatrices1el, edgecolor='black', linewidth=1.2)
    plt.show()
    return fileName
###END OF ONE LAYER###


def logAndCorrelation2(element1, element2, row, column, bins):
    element1Value = ""
    element2Value = ""

    for key in element1.keys():
        if key != 'Lat' and key!='Long':
            element1Value = key
    for key in element2.keys():
        if key != 'Lat' and key!='Long':
            element2Value = key

    fileName = element1Value + element2Value + str(row*column) +'SubsectionsLogAndCorrelationPlot.png'
    if os.path.isfile(fileName):
        return fileName

    xRange_e1 = element1['Lat'].max() - element1['Lat'].min()
    yRange_e1 = element1['Long'].max() - element1['Long'].min()
    xRange_e2 = element2['Lat'].max() - element2['Lat'].min()
    yRange_e2 = element2['Long'].max() - element2['Long'].min()

    section_e1 = math.ceil(xRange_e1/row)         # length specified by user
    ySection_e1 = math.ceil(yRange_e1/column)
    section_e2 = math.ceil(xRange_e2/row)
    ySection_e2 = math.ceil(yRange_e2/column)

    min_e1 = element1['Lat'].min()
    max_e1 = element1['Lat'].max()
    min_e2 = element2['Lat'].min()
    max_e2 = element2['Lat'].max()
    ymin_e1 = element1['Long'].min()
    ymax_e1 = element1['Long'].max()
    ymin_e2 = element2['Long'].min()
    ymax_e2 = element2['Long'].max()

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

    e2_d={}
    for x in range(0,row):
        for y in range(0,column):
            e2_d["matrix{0}{1}".format(x,y)]=element2[(min_e2+x*section_e2 <= element2['Lat']) & (element2['Lat'] < min_e2+(x+1)*section_e2) & (ymin_e2+y*ySection_e2 <= element2['Long']) & (element2['Long'] < ymin_e2+(y+1)*ySection_e2)]

    e2_log_arr = []
    for key, value in e2_d.items():
        e2_key_std = numpy.std(e2_d[key][element2Value])
        e2_key_log = numpy.log(e2_key_std)
        e2_log_arr.append(e2_key_log)

    # plt(x,y) -> plt(e2, e1) since first element should be on y axis and second element on x axis based on visualization paper
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.title(r'$log(\sigma_1)\ vs\ \log(\sigma_2)$')
    plt.xlabel(r'$('+element2Value+')\ \log(\sigma_2)$')
    plt.ylabel(r'$('+element1Value+')\ \log(\sigma_1)$')
    plt.scatter(e2_log_arr, e1_log_arr)
    plt.tight_layout()

    ##### Correlation part #####
    el1 = element1.corr()
    el2 = element2.corr()

    first_el = el1[element1Value]
    second_el = el2[element2Value]

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
    plt.subplot(2, 1, 2)
    plt.title(r'$log(\sigma_1)\ vs\ \rho_{12}$')
    plt.xlabel(r'$\rho_{12}$')
    plt.ylabel(r'$('+ element1Value +')\ \log(\sigma_1)$')
    plt.scatter(corr_d, e1_log_arr)
    plt.tight_layout()
    #plt.show()
    #Histograms
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title(r'$Frequency\ vs\ log(\sigma_1)$')
    plt.xlabel(r'$log(\sigma_1)$')
    plt.ylabel(r'Frequency')
    plt.hist(e1_log_arr, bins, edgecolor='black', linewidth=1.2)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.title(r'$Frequency\ vs\ \rho_{12}$')
    plt.xlabel(r'$\rho_{12}$')
    plt.ylabel(r'Frequency')
    plt.hist(corr_d, bins, edgecolor='black', linewidth=1.2)
    plt.tight_layout()
    plt.show()
###END OF TWO LAYER###
