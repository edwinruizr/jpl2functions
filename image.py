from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import pyproj
import struct
import numpy
import sys
import numpy.ma as ma
import logging
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
    print("gdalfile", gdalfile)
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
def getXCoordinate(indexX):
    return (indexX/128)-180


def getYCoordinate(indexY):
    return (indexY/128)-90*-1


def printFile(openedGDALfile):
    rasterBand = openedGDALfile.GetRasterBand(1)
    data_raster = rasterBand.ReadAsArray().astype(numpy.float)
    print(data_raster)
    print(len(data_raster))
    print(len(data_raster[0]))
    print(data_raster[2][1])

#
#
#FUNCTION DEFINITIONS - END

# print("LRO_NAC_Slope_15m_20N010E_2mp.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
# statsArray = getStatsArray("LRO_NAC_Slope_15m_20N010E_2mp.tif")
# print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

# print("LP_GRS_Fe_Global_2ppd.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
# statsArray = getStatsArray("LP_GRS_Fe_Global_2ppd.tif")
# print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

# print("LP_GRS_H_Global_2ppd.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
# statsArray = getStatsArray("LP_GRS_H_Global_2ppd.tif")
# print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

# print("LP_GRS_K_Global_halfppd.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
# statsArray = getStatsArray("LP_GRS_K_Global_halfppd.tif")
# print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

# print("LP_GRS_Th_Global_2ppd.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
# statsArray = getStatsArray("LP_GRS_Th_Global_2ppd.tif")
# print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

# print("LRO_LOLA_DEM_Global_128ppd_v04.tif\naverage \t\tmean \t\tmedian \t\tstd deviation \t\tvariance")
# statsArray = getStatsArray("LRO_LOLA_DEM_Global_128ppd_v04.tif")
# print("{} \t\t{} \t{} \t{} \t\t{}".format(statsArray[0],statsArray[1],statsArray[2],statsArray[3],statsArray[4]))

# OPEN TIF FILES & CONVERT THEM TO NUMPY ARRAYS
# slope tif
slopeTifFile = gdal.Open( "LRO_NAC_Slope_15m_20N010E_2mp.tif", gdal.GA_ReadOnly )
slopeNumpyArray = numpy.array(slopeTifFile.ReadAsArray())               # converts file opened to a numpy array

# iron tif
feTiffFile = gdal.Open("LP_GRS_Fe_Global_2ppd.tif",gdal.GA_ReadOnly)
feNumpyArray = numpy.array(feTiffFile.ReadAsArray())                    # converts file opened to a numpy array

# # LOLA DEM tif
# lolademFile = gdal.Open("LRO_LOLA_DEM_Global_128ppd_v04.tif", gdal.GA_ReadOnly)
# lolaNumpyArray = numpy.array(lolademFile.ReadAsArray()).astype(numpy.float)

# # helium tif
# hTifFile = gdal.Open("LP_GRS_H_Global_2ppd.tif", gdal.GA_ReadOnly)
# hNumpyArray = numpy.array(hTifFile.ReadAsArray())

# # potassium tif
# kTifFile = gdal.Open("LP_GRS_K_Global_halfppd.tif", gdal.GA_ReadOnly)
# kNumpyArray = numpy.array(kTifFile.ReadAsArray())

# # thorium tif
# thTifFile = gdal.Open("LP_GRS_Th_Global_2ppd.tif", gdal.GA_ReadOnly)
# thNumpyArray  = numpy.array(thTifFile.ReadAsArray())


# get origin and pixel size of slope tif file
print('LRO_NAC_Slope_15m_20N010E_2mp.tif')
geotransform = slopeTifFile.GetGeoTransform()
if geotransform:
    print("Origin of slope tiff = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

# calculate max, min, and average for slope tif file
arrayOfIndexes = getMaxIndex2d(slopeTifFile)
arrayOfMinIndexes = getMinIndex2d(slopeTifFile)

print ('Max value is at index [{}][{}] = {}'.format(arrayOfIndexes[0][0],arrayOfIndexes[1][0],slopeNumpyArray[arrayOfIndexes[0][0]][arrayOfIndexes[1][0]]))
print()
print ('Min value is at index [{}][{}] = {}'.format(arrayOfMinIndexes[0][0], arrayOfMinIndexes[1][0],slopeNumpyArray[arrayOfMinIndexes[0][0]][arrayOfMinIndexes[1][0]]))
print()
print(geotransform[2])
print(geotransform[0])
print(geotransform[3])
print("pixel size" ,geotransform[1])
print("pixel size ", geotransform[5])
print("raster x size", slopeTifFile.RasterXSize)
printFile(slopeTifFile)

# # get origin and pixel size of fe tif file
# print('LP_GRS_Fe_Global_2ppd.tif')
# transform = feTiffFile.GetGeoTransform()
# if transform:
#     print("Origin of FE tiff file = ({}, {})".format(transform[0], transform[3]))
#     print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# # calculate max, min, and average for fe tif file
# indexOfMax = getMaxIndex2d(feTiffFile)
# logging.debug('type of indexOfMax {}'.format(type(indexOfMax)))
# indexOfMin = getMinIndex2d(feTiffFile)
# logging.debug('type of indexOfMin {}'.format(type(indexOfMin)))
# print("Max value of Fe Global tiff file is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], feNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
# print('Min value of Fe Global tiff file is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], feNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))

# # get origin and pixel size of lola dem tif file
# print('LRO_LOLA_DEM_Global_128ppd_v04.tif')
# transform = lolademFile.GetGeoTransform()   # overwrite our old variable transform
# if transform:
#     print("Origin of lola dem tiff file = ({}, {})".format(transform[0], transform[3]))
#     print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# # calculate max, min, and average for lola dem tif file
# indexOfMax = getMaxIndex2d(lolademFile)
# indexOfMin = getMinIndex2d(lolademFile)
# print("Max value of lola dem tiff file is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], lolaNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
# print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], lolaNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
# print("Projection is {}".format(lolademFile.GetProjection()))
# x = getXCoordinate(indexOfMax[0][0])
# y = getYCoordinate(indexOfMax[1][0])
# print('max x coordinate : {}'.format(x))
# print('max y coordinate : {}'.format(y))



# # get origin and pixel size of H tif file
# print('LP_GRS_H_Global_2ppd.tif')
# transform = hTifFile.GetGeoTransform()   # overwrite our old variable transform
# if transform:
#     print("Origin of H tiff file = ({}, {})".format(transform[0], transform[3]))
#     print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# # calculate max, min, and average for H tif file
# indexOfMax = getMaxIndex2d(hTifFile)
# indexOfMin = getMinIndex2d(hTifFile)

# print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], hNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
# print("Min value is at index [{}][{}] = {}".format(indexOfMin[0][0], indexOfMin[1][0], hNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))


# # get origin and pixel size of K tif file
# print('LP_GRS_K_Global_halfppd.tif')
# transform = kTifFile.GetGeoTransform()   # overwrite our old variable transform
# if transform:
#     print("Origin of K tiff file = ({}, {})".format(transform[0], transform[3]))
#     print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# # calculate max, min, and average for K tif file
# indexOfMax = getMaxIndex2d(kTifFile)
# indexOfMin = getMinIndex2d(kTifFile)
# print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], kNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
# print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], kNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))

# # get origin and pixel size of Th tif file
# print('LP_GRS_Th_Global_2ppd.tif')
# transform = thTifFile.GetGeoTransform()   # overwrite our old variable transform
# if transform:
#     print("Origin of Th tif file = ({}, {})".format(transform[0], transform[3]))
#     print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# # calculate max, min, and average for Th tif file
# indexOfMax = getMaxIndex2d(thTifFile)
# indexOfMin = getMinIndex2d(thTifFile)
# print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], thNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
# print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], thNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
