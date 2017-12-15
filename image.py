from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
from pyproj import Proj, transform
import struct
import numpy
import sys
import numpy.ma as ma
import logging
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.axes as ax

import pandas as pd
import geopandas as gpd
import georasters as gr


logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s- %(message)s')
logging.debug('Start of program')
# FUNCTION DEFINITIONS - START
#
#
# function that returns an array of average, mean, median, standard deviation and variance in that order
# the input is the tif File
# Ex. getStatsArray("LRO_LOLA_DEM_Global_128ppd_v04.tif")

def getStatsArray(tifFileName):
    # Open data
    gdalfile = gdal.Open(tifFileName)
    logging.debug("gdalfile var is of type {}".format(type(gdalfile)))
    # open the first raster band (in the case of the data tif files the only band)
    banddataraster = gdalfile.GetRasterBand(1)
    logging.debug("banddataraster var is of type {}".format(type(banddataraster)))
    # Read raster as arrays
    dataraster = banddataraster.ReadAsArray().astype(numpy.float)
    logging.debug("dataraster var is of type {}".format(type(dataraster)))
    # convert raster array to masked array disregarding the no data value in raster
    numpymaskedarray = numpy.ma.masked_array(dataraster,  dataraster == banddataraster.GetNoDataValue())
    logging.debug("numpymaskedarray var is of type {}".format(type(numpymaskedarray)))
    # Calculate statistics of zonal raster and return them in an array
    return numpy.average(numpymaskedarray),numpy.mean(numpymaskedarray),numpy.median(numpymaskedarray),numpy.std(numpymaskedarray),numpy.var(numpymaskedarray)

### function that returns index of max value
### in the form of 2d array
def getMaxIndex2d(openedGDALfile):
    # openedGDALfile is what gets returned when you do gdal.Open()
    rasterBand = openedGDALfile.GetRasterBand(1)
    # convert raster band to array
    dataraster = rasterBand.ReadAsArray().astype(numpy.float)
    # convert to masked array. the raster band has a function that returns the no data value
    numpymaskedarray = numpy.ma.masked_array(dataraster, dataraster == rasterBand.GetNoDataValue())
    # return where max occurs
    return numpy.where(numpymaskedarray == numpy.max(numpymaskedarray))

### function that returns index of min value
### same thing as getMaxIndex2d function above
def getMinIndex2d(openedGDALfile):
    rasterBand = openedGDALfile.GetRasterBand(1)
    noValue = rasterBand.GetNoDataValue()
    numpyTiffArray = numpy.array(openedGDALfile.ReadAsArray())
    tiffArray = ma.masked_array(numpyTiffArray, numpyTiffArray == noValue)
    return numpy.where(tiffArray == numpy.min(tiffArray))

### my functions that got the relative location of point
### need to be replace by function that converts to longitude latitude coordinates
def getXCoordinate(indexX, pixel_x, upper_left_lat):
    return upper_left_lat+indexX*pixel_x

def getYCoordinate(indexY, pixel_y, upper_left_lon):
    return upper_left_lon-indexY*pixel_y


def convertToLatLong(coordX,coordY, gdalFile):
    inSRS_wkt = gdalFile.GetProjection()  # gives SRS in WKT
    logging.debug(inSRS_wkt)
    inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
    inSRS_converter.ImportFromWkt(inSRS_wkt)  # populates the spatial ref object with our WKT SRS
    inSRS_forPyProj = inSRS_converter.ExportToProj4()  # Exports an SRS ref as a Proj4 string usable by PyProj
    inProj = Proj(inSRS_forPyProj)
    logging.debug(inSRS_forPyProj)
    outProj = Proj(init='epsg:4326')
    return transform(inProj,outProj,coordX,coordY)

def convertToDecDeg(coordX,coordY, gdalFile):
    inSRS_wkt = gdalFile.GetProjection()  # gives SRS in WKT
    logging.debug(inSRS_wkt)
    inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
    inSRS_converter.ImportFromWkt(inSRS_wkt)  # populates the spatial ref object with our WKT SRS
    inSRS_forPyProj = inSRS_converter.ExportToProj4()  # Exports an SRS ref as a Proj4 string usable by PyProj
    inProj = Proj(init='epsg:4326')
    outProj = Proj(inSRS_forPyProj)
    logging.debug(inSRS_forPyProj)
    return transform(inProj,outProj,coordX,coordY)


    
#
#
#FUNCTION DEFINITIONS - END

#print("LRO_NAC_Slope_15m_20N010E_2mp.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
#statsArray = getStatsArray("LRO_NAC_Slope_15m_20N010E_2mp.tif")
#print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

#print("LP_GRS_Fe_Global_2ppd.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
#statsArray = getStatsArray("LP_GRS_Fe_Global_2ppd.tif")
#print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

#print("LP_GRS_H_Global_2ppd.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
#statsArray = getStatsArray("LP_GRS_H_Global_2ppd.tif")
#print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

#print("LP_GRS_K_Global_halfppd.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
#statsArray = getStatsArray("LP_GRS_K_Global_halfppd.tif")
#print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

#print("LP_GRS_Th_Global_2ppd.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
#statsArray = getStatsArray("LP_GRS_Th_Global_2ppd.tif")
#print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

#print("LRO_LOLA_DEM_Global_128ppd_v04.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
#statsArray = getStatsArray("LRO_LOLA_DEM_Global_128ppd_v04.tif")
#print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

# OPEN TIF FILES & CONVERT THEM TO NUMPY ARRAYS OR DATAFRAMES
# slope tif
tif_file = "LRO_NAC_Slope_15m_20N010E_2mp.tif"
tif_file2 = "LP_GRS_Th_Global_2ppd.tif"
#xyz_file
xyz_file = "temp_avg_hour00.xyz"
# slopeTifFile = gdal.Open( tif_file, gdal.GA_ReadOnly )
# slopeTifFile = gdal.Open( tif_file2, gdal.GA_ReadOnly )
#slopeNumpyArray = numpy.array(slopeTifFile.ReadAsArray())               # converts file opened to a numpy array

# #reads it in as georasters then convert it to pandas
# stuff = gr.from_file(tif_file)
# df = stuff.to_pandas()
# print("slope dataframe.head and shape")
# print(df.head())
# print(df.shape)

# stuff2 = gr.from_file(tif_file2)
# df2 = stuff2.to_pandas()
# print("th global dataframe and shape")
# print(df2.head())
# print(df2.shape)

#reads in xyz file
xyz_file = numpy.loadtxt(xyz_file, dtype= 'str')
names = ['Longtitude', 'Latitude', 'Temp(k)']
xyz_dataframe = pd.DataFrame(xyz_file)
xyz_dataframe.columns = names

print("xyz dataframe and shape")
print(xyz_dataframe.head())
print(xyz_dataframe.shape)
temp = xyz_dataframe['Temp(k)']
print(temp[:5])




# df = gpd.read_file(slopeNumpyArray)

# #print first 5 lines in data frame

# print("slope\n", df.head())

# print("th global\n", df2.head())


#Prints the dataframe value every 10000
#print(df['value'][::10000])


# print("slope numpy array", slopeNumpyArray)
# print(slopeNumpyArray.shape)
# print("type of file", type(slopeTifFile))
# print("type of file for numpy array", type(slopeNumpyArray))
# print("max", numpy.max(slopeNumpyArray))

# temp_slope = slopeNumpyArray.flatten()
# print(len(temp_slope))
# n, bins, patches = plt.hist(temp_slope, 50, normed = 1, facecolor = 'blue')
# plt.title('Histogram for Slope')
# plt.xlabel('Frequency')
# plt.ylabel('value')
# plt.grid(True)
# plt.show()


# iron tif
#feTiffFile = gdal.Open("LP_GRS_Fe_Global_2ppd.tif",gdal.GA_ReadOnly)
#feNumpyArray = numpy.array(feTiffFile.ReadAsArray())                    # converts file opened to a numpy array

# LOLA DEM tif
#lolademFile = gdal.Open("LRO_LOLA_DEM_Global_128ppd_v04.tif", gdal.GA_ReadOnly)
#lolaNumpyArray = numpy.array(lolademFile.ReadAsArray()).astype(numpy.float)

# helium tif
#hTifFile = gdal.Open("LP_GRS_H_Global_2ppd.tif", gdal.GA_ReadOnly)
#hNumpyArray = numpy.array(hTifFile.ReadAsArray())

# potassium tif
#kTifFile = gdal.Open("LP_GRS_K_Global_halfppd.tif", gdal.GA_ReadOnly)
#kNumpyArray = numpy.array(kTifFile.ReadAsArray())

# thorium tif
#thTifFile = gdal.Open("LP_GRS_Th_Global_2ppd.tif", gdal.GA_ReadOnly)
#thNumpyArray  = numpy.array(thTifFile.ReadAsArray())








# get origin and pixel size of slope tif file
# print('LRO_NAC_Slope_15m_20N010E_2mp.tif')
# geotransform = slopeTifFile.GetGeoTransform()

# if geotransform:
#     print("Origin of slope tiff = ({}, {})".format(geotransform[0], geotransform[3]))
#     print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

# # calculate max, min, and average for slope tif file
# arrayOfIndexes = getMaxIndex2d(slopeTifFile)
# arrayOfMinIndexes = getMinIndex2d(slopeTifFile)

# print ('Max value is at index [{}][{}] = {}'.format(arrayOfIndexes[0][0],arrayOfIndexes[1][0],slopeNumpyArray[arrayOfIndexes[0][0]][arrayOfIndexes[1][0]]))
# print ('Min value is at index [{}][{}] = {}'.format(arrayOfMinIndexes[0][0], arrayOfMinIndexes[1][0],slopeNumpyArray[arrayOfMinIndexes[0][0]][arrayOfMinIndexes[1][0]]))

# # test coordinate getter
# #returns upper left (0,0)
# print (getXCoordinate(0, geotransform[1], geotransform[0]))
# print (getYCoordinate(0, geotransform[5], geotransform[3]))

# #returns lower right (5286,14695)
# print (getXCoordinate(5286, geotransform[1], geotransform[0]))
# print (getYCoordinate(14695, geotransform[5], geotransform[3]))

# convertToLatLong(geotransform[0], geotransform[3],slopeTifFile)

# x2,y2 = convertToLatLong(-4838396, 622172, slopeTifFile)
# print("({},{})".format(x2,y2))







# get origin and pixel size of fe tif file
#print('LP_GRS_Fe_Global_2ppd.tif')
#transform = feTiffFile.GetGeoTransform()
#if transform:
#    print("Origin of FE tiff file = ({}, {})".format(transform[0], transform[3]))
#    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for fe tif file
#indexOfMax = getMaxIndex2d(feTiffFile)
#logging.debug('type of indexOfMax {}'.format(type(indexOfMax)))
#indexOfMin = getMinIndex2d(feTiffFile)
#logging.debug('type of indexOfMin {}'.format(type(indexOfMin)))
#print("Max value of Fe Global tiff file is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], feNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
#print('Min value of Fe Global tiff file is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], feNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))

# get origin and pixel size of lola dem tif file
#print('LRO_LOLA_DEM_Global_128ppd_v04.tif')
#transform = lolademFile.GetGeoTransform()   # overwrite our old variable transform
#if transform:
#    print("Origin of lola dem tiff file = ({}, {})".format(transform[0], transform[3]))
#    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for lola dem tif file
#indexOfMax = getMaxIndex2d(lolademFile)
#indexOfMin = getMinIndex2d(lolademFile)
#print("Max value of lola dem tiff file is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], lolaNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
#print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], lolaNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
#print("Projection is {}".format(lolademFile.GetProjection()))
#x = getXCoordinate(indexOfMax[0][0])
#y = getYCoordinate(indexOfMax[1][0])
#print('max x coordinate : {}'.format(x))
#print('max y coordinate : {}'.format(y))


# get origin and pixel size of H tif file
#print('LP_GRS_H_Global_2ppd.tif')
#transform = hTifFile.GetGeoTransform()   # overwrite our old variable transform
#if transform:
#    print("Origin of H tiff file = ({}, {})".format(transform[0], transform[3]))
#    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for H tif file
#indexOfMax = getMaxIndex2d(hTifFile)
#indexOfMin = getMinIndex2d(hTifFile)

#print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], hNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
#print("Min value is at index [{}][{}] = {}".format(indexOfMin[0][0], indexOfMin[1][0], hNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))

# get origin and pixel size of K tif file
#print('LP_GRS_K_Global_halfppd.tif')
#transform = kTifFile.GetGeoTransform()   # overwrite our old variable transform
#if transform:
#    print("Origin of K tiff file = ({}, {})".format(transform[0], transform[3]))
#    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for K tif file
#indexOfMax = getMaxIndex2d(kTifFile)
#indexOfMin = getMinIndex2d(kTifFile)
#print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], kNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
#print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], kNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))

# get origin and pixel size of Th tif file
#print('LP_GRS_Th_Global_2ppd.tif')
#transform = thTifFile.GetGeoTransform()   # overwrite our old variable transform
#if transform:
#    print("Origin of Th tif file = ({}, {})".format(transform[0], transform[3]))
#    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for Th tif file
#indexOfMax = getMaxIndex2d(thTifFile)
#indexOfMin = getMinIndex2d(thTifFile)
#print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], thNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
#print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], thNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
