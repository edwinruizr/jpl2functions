import inquirer
from functionsLib import *

## FUNCTIONS
# subsamples dataframe by percent, ie subsamplePercent = 5 means 5% of values in dataframe
def dataframeSubsampler(df,subsamplePercent):
    subsampleVal = 100/subsamplePercent
    df_out = df[0::int(subsampleVal)]
    return df_out


# TODO - figure how to implement this with the choices
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



#VISUALIZATION
###ONE LAYER VISUAL###
def logAndCorrelation1(element1, rows, column, bins):
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
    #Histogram
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.title(r'$Frequency\ vs\ log(\sigma_1)$')
    plt.xlabel(r'$log(\sigma_1)$')
    plt.ylabel(r'Frequency')
    plt.hist(e1_log_arr, bins)
    #Variance
    e1_Var = numpy.var(e1_d[key][element1Value])
    plt.subplot(2,2,2)
    plt.title(r'$Frequency\ vs\ Effective\ Variance$')
    plt.xlabel(r'$Effective Variance$')
    plt.ylabel(r'Frequency')
    plt.hist(e1_Var, bins)
    plt.show()
    return fileName
###END OF ONE LAYER###
###TWO LAYER VISUAL###
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
    plt.subplot(2,2,1)
    plt.title(r'$log(\sigma_1)\ vs\ \log(\sigma_2)$')
    plt.xlabel(r'$('+element2Value+')\ \log(\sigma_2)$')
    plt.ylabel(r'$('+element1Value+')\ \log(\sigma_1)$')
    plt.scatter(e2_log_arr, e1_log_arr)

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
    plt.subplot(2,2,2)
    plt.title(r'$log(\sigma_1)\ vs\ \rho_{12}$')
    plt.xlabel(r'$\rho_{12}$')
    plt.ylabel(r'$('+ element1Value +')\ \log(\sigma_1)$')
    plt.scatter(corr_d, e1_log_arr)

    #Histograms
    plt.subplot(2,2,4)
    plt.title(r'$Frequency\ vs\ \rho_{12}$')
    plt.xlabel(r'$\rho_{12}$')
    plt.ylabel(r'Frequency')
    plt.hist(corr_d, bins)
    plt.show()
    return fileName
###END OF TWO LAYER###


###THREE LAYER VISUAL###
def logAndCorrelation3(element1, element2, element3, row, column, bins):
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
    plt.subplot(4,3,2)
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

    plt.subplot(4,3,5)
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
    #p13 correlation
    correlation3 = numpy.corrcoef(el1, el3)
    corr_d3=[]
    for x in range(0, len(correlation2)):
        for y in range(0, len(correlation3)):
            corr_d3.append(correlation2[x][y])

    #Graph to show element1 vs p23
    plt.subplot(4,3,8)
    plt.title(r'$log(\sigma_1)\ vs\ \rho_{23}$')
    plt.xlabel(r'$\rho_{23}$')
    plt.ylabel(r'$('+ element1Value +')\ \log(\sigma_1)$')
    plt.scatter(corr_d2, e1_log_arr)

    #Graph to show p12 vs p23
    plt.subplot(4,3,11)
    plt.title(r'$\rho_{12}\ vs\ \rho_{23}$')
    plt.xlabel(r'$\rho_{23}$')
    plt.ylabel(r'$\rho_{12}$')
    plt.scatter(corr_d2, corr_d)

    plt.subplot(4,3,1)
    plt.title(r'$Frequency\ vs\ \rho_{12}$')
    plt.xlabel(r'$\rho_{12}$')
    plt.ylabel(r'Frequency')
    plt.hist(corr_d, bins)
    plt.subplot(4,3,4)
    plt.title(r'$Frequency\ vs\ \rho_{23}$')
    plt.xlabel(r'$\rho_{23}$')
    plt.ylabel(r'Frequency')
    plt.hist(corr_d2, bins)

    fig = plt.figure(2)
    ax = Axes3D(fig)
    ax.scatter(corr_d, corr_d2, corr_d3)
    ax.set_xlabel(r'$\rho_{12}$')
    ax.set_ylabel(r'$\rho_{23}$')
    ax.set_zlabel(r'$\rho_{13}$')
    plt.show()
    return fileName
###END OF THREE LAYER###


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
    tempDf = fileToDataframe(names[answer], answer)
    # TODO - call conversion function HERE
    tempDf = tempDf.rename(columns={"x": "Lat", "y": "Long"})
    df.append(tempDf)
#df now is a list of dataframes for the files selected


if len(df) == 2:
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Plot x vs y', 'Visualization Graphs(2)']
# chose multiple layers442
elif len(df) > 1:
    choicesList = ['Stats', 'Covariance', 'Correlation', 'Clustering', 'Visualization Graphs(3)']
# didn't choose any
elif len(df) == 0:
    print("You didn't choose any layers. Exiting.")
    exit(0)
# they chose 1 layer
else:
    choicesList = ['Stats', 'Variance', 'Histogram', 'Plot layer', '3d plot', 'Clustering', 'Visualization Graphs(1)']


analysis = [
             inquirer.Checkbox('Analysis',
                               message="What kinds of analysis do you want to run on the layers chosen?",
                               choices= choicesList,
                               ),
             ]
respuesta = inquirer.prompt(analysis)
dataframe = aggregateValues(df)

#done
if 'Stats' in respuesta['Analysis']:
    print(getStats(dataframe))

#done
if 'Covariance' in respuesta['Analysis']:
    aggregatedDataframe = aggregateValues(df)
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            Image.open(visualizeCovariance(dataframe, answers['Layers'], norm = True)).show()
            break
        elif visualizeInput == 'n':
            Image.open(visualizeCovariance(dataframe, answers['Layers'], norm = False)).show()
            break

#done
if 'Correlation' in respuesta['Analysis']:
    # normalization doesn't matter (produces the same output)
    img = Image.open(visualizeCorrelation(dataframe, answers['Layers']))
    img.show()

#done
if 'Variance' in respuesta['Analysis']:
    img = Image.open(visualizeVariance(dataframe, answers['Layers'][0]))
    img.show()

#done
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
    fileName = plotHistogram(dataframe, bins)
    Image.open(fileName).show()




###ONE LAYER VISUAL###
if 'Visualization Graphs(1)' in respuesta['Analysis']:
    # To show up the first time
    row = int(input("How many rows for matrix? "))
    column = int(input("How many columns for matrix? "))
    bins = int(input("How many bins? "))
    logAndCorrelation1(df[0], row, column, bins)

    # To allow user to change the size of the length without having to reload the program each time (Dr. Zhu wanted to implement this)
    # Also to allow user to exit if they want to stop looking at graphs
    while True:
        userChoice = input("\nWould you like to continue to view graphs? (y/n)")
        if userChoice == 'y':
            row = int(input("How many rows for matrix? "))
            column = int(input("How many columns for matrix? "))
            bins = int(input("How many bins? "))
            fileName = logAndCorrelation1(df[0], row, column, bins)
            Image.open(fileName).show()
        elif userChoice == 'n':
            exit(0)
###END OF ONE LAYER VISUAL###

###TWO LAYERS VISUAL###
if 'Visualization Graphs(2)' in respuesta['Analysis']:
    # To show up the first time
    row = int(input("How many rows for matrix? "))
    column = int(input("How many columns for matrix? "))
    bins = int(input("How many bins? "))
    logAndCorrelation2(df[0], df[1], row, column, bins)

    # To allow user to change the size of the length without having to reload the program each time (Dr. Zhu wanted to implement this)
    # Also to allow user to exit if they want to stop looking at graphs
    while True:
        userChoice = input("\nWould you like to continue to view graphs? (y/n)")
        if userChoice == 'y':
            row = int(input("How many rows for matrix? "))
            column = int(input("How many columns for matrix? "))
            bins = int(input("How many bins? "))
            fileName = logAndCorrelation2(df[0], df[1], row, column, bins)
            Image.open(fileName).show()
        elif userChoice == 'n':
            exit(0)
##END OF TWO LAYER VISUAL##

###THREE LAYER VISUAL###
if 'Visualization Graphs(3)' in respuesta['Analysis']:
    # To show up the first time
    row = int(input("How many rows for matrix? "))
    column = int(input("How many columns for matrix? "))
    bins = int(input("How many bins? "))
    logAndCorrelation3(df[0], df[1], df[2], row, column, bins)

    # To allow user to change the size of the length without having to reload the program each time (Dr. Zhu wanted to implement this)
    # Also to allow user to exit if they want to stop looking at graphs
    while True:
        userChoice = input("\nWould you like to continue to view graphs? (y/n)")
        if userChoice == 'y':
            row = int(input("How many rows for matrix? "))
            column = int(input("How many columns for matrix? "))
            bins = int(input("How many bins? "))
            fileName = logAndCorrelation3(df[0], df[1], df[2], row, column, bins)
            Image.open(fileName).show()
        elif userChoice == 'n':
            exit(0)
###END OF THREE LAYER VISUAL###

# clustering
# TODO NEEDS TO TAKE IN SOME OTHER STUFF, LIKE COLUMN NAMES, NEED TO BE ABLE TO READ IN FUTURE DATAFRAMES
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
            visualizeInput = input("Do you want to normalize "+ answers['Layers'][i] +" data? (y/n)")
            if visualizeInput == 'y':
                normalizeBoolList.append(True)
                break
            elif visualizeInput == 'n':
                normalizeBoolList.append(False)
                break


    wholeDf = df[0]
    for x in range(0, len(df)-1):
        wholeDf = pandas.merge(wholeDf, df[x+1], how='inner', on=['Lat', 'Long'])

    for answer in answers['Layers']:
        wholeDf=wholeDf[wholeDf[answer].notnull()]

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
        elementsList.append(layer)
    labels = kmean(wholeDf, elementsList, normalizeBoolList, clusterSize)
    # run function with choices choosen
    if cAnswerLength == 2:
        kmean_plot_2_val(wholeDf, cAnswer['Cluster'], labels, clusterSize)

    if cAnswerLength == 3:
        make3dClusterPlot(dataframe_label_assign(wholeDf,cAnswer['Cluster'],labels),color_array)


if 'Plot x vs y' in respuesta['Analysis']:
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            normalize = True
            break
        elif visualizeInput == 'n':
            normalize = False
            break

    while True:
        visualizeInput = input("Do you want the axes to be of equal aspect? (y/n)")
        if visualizeInput == 'y':
            equalAspect = True
            break
        elif visualizeInput == 'n':
            equalAspect = False
            break

    while True:
        try:
            sample = int(input("Enter number of random sample from data. (min: 200 - " + str(dataframe.shape[0]) + ")"))
        except ValueError:
            print("Enter a number!")
            sample = 0
        if 200 <= sample <= dataframe.shape[0]:
            break

    fileName = plot_x_y_normalize_all(dataframe.sample(n=sample), answers['Layers'][0], answers['Layers'][1], normalize, equalAspect, 0.5)
    Image.open(fileName).show()


if 'Plot layer' in respuesta['Analysis']:
    while True:
        visualizeInput = input("Do you want to normalize the data? (y/n)")
        if visualizeInput == 'y':
            fileName = plot_3_val_normalize_all(df[0], 'Lat', 'Long', answers['Layers'][0], True, 1)
            Image.open(fileName).show()
            break
        elif visualizeInput == 'n':
            fileName = plot_3_val_normalize_all(df[0], 'Lat', 'Long', answers['Layers'][0], False, 1)
            Image.open(fileName).show()
            break


if '3d plot' in respuesta['Analysis']:
    make3dPlot(df[0], 'Lat', 'Long', answers['Layers'][0], 'blue' , 0.2)



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

        subsampleDF = dataframeSubsampler(completeDataframe,5)
        plotAllTemp(subsampleDF,'Longitude','Latitude',subsampleDF.columns[2:26],color_array)
        break
    elif visualizeTemp == 'n':
        print("bless your soul!")
        break
