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
# FUNCTION DEFINITIONS - START
#
#
#function that returns index of max value
def getMaxIndex2d(openedGDALfile):                                          # openedGDALfile is what gets returned when you do gdal.Open()
    rasterBand = openedGDALfile.GetRasterBand(1)                            # selects the first band (in this case the only band)
    noValue = rasterBand.GetNoDataValue()                                   # the raster band has a function that returns the no data value
    numpyTiffArray = numpy.array(openedGDALfile.ReadAsArray())              # convert the object returned from gdal.Open() to a numpy array
    tiffArray = ma.masked_array(numpyTiffArray, numpyTiffArray == noValue)  # convert the numpy array to a masked array. Second parameter is the no data value so that it disregards it
    return numpy.where(tiffArray == numpy.max(tiffArray))                   # masked array does not include the no data value into calculations so it's safe (and i tested it) to use the numpy.max function

#function that returns index of min value
def getMinIndex2d(openedGDALfile):
    rasterBand = openedGDALfile.GetRasterBand(1)
    noValue = rasterBand.GetNoDataValue()
    numpyTiffArray = numpy.array(openedGDALfile.ReadAsArray())
    tiffArray = ma.masked_array(numpyTiffArray, numpyTiffArray == noValue)
    return numpy.where(tiffArray == numpy.min(tiffArray))

#function that returns mean value
def getAverage(openedGDALfile):
    rasterBand = openedGDALfile.GetRasterBand(1)
    noValue = rasterBand.GetNoDataValue()
    numpyTiffArray = numpy.array(openedGDALfile.ReadAsArray())
    tiffArray = ma.masked_array(numpyTiffArray, numpyTiffArray == noValue)
    return tiffArray.mean()

def getXCoordinate(indexX):
    return (indexX/128)-180


def getYCoordinate(indexY):
    return (indexY/128)-90*-1

#
#
#FUNCTION DEFINITIONS - END

# OPEN TIF FILES & CONVERT THEM TO NUMPY ARRAYS
# slope tif
slopeTifFile = gdal.Open( "LRO_NAC_Slope_15m_20N010E_2mp.tif", gdal.GA_ReadOnly )
slopeNumpyArray = numpy.array(slopeTifFile.ReadAsArray())               # converts file opened to a numpy array

# iron tif
feTiffFile = gdal.Open("LP_GRS_Fe_Global_2ppd.tif",gdal.GA_ReadOnly)
feNumpyArray = numpy.array(feTiffFile.ReadAsArray())                    # converts file opened to a numpy array

# LOLA DEM tif
lolademFile = gdal.Open("LRO_LOLA_DEM_Global_128ppd_v04.tif", gdal.GA_ReadOnly)
lolaNumpyArray = numpy.array(lolademFile.ReadAsArray())

# helium tif
hTifFile = gdal.Open("LP_GRS_H_Global_2ppd.tif", gdal.GA_ReadOnly)
hNumpyArray = numpy.array(hTifFile.ReadAsArray())

# potassium tif
kTifFile = gdal.Open("LP_GRS_K_Global_halfppd.tif", gdal.GA_ReadOnly)
kNumpyArray = numpy.array(kTifFile.ReadAsArray())

# thorium tif
thTifFile = gdal.Open("LP_GRS_Th_Global_2ppd.tif", gdal.GA_ReadOnly)
thNumpyArray  = numpy.array(thTifFile.ReadAsArray())


# get origin and pixel size of slope tif file
print('LRO_NAC_Slope_15m_20N010E_2mp.tif')
geotransform = slopeTifFile.GetGeoTransform()
if geotransform:
    print("Origin of slope tiff = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

# calculate max, min, and average for slope tif file
arrayOfIndexes=getMaxIndex2d(slopeTifFile)
arrayOfMinIndexes = getMinIndex2d(slopeTifFile)

#print ('Max value is at index [', arrayOfIndexes[0][0], '][', arrayOfIndexes[1][0], '] = ', slopeNumpyArray[arrayOfIndexes[0][0]][arrayOfIndexes[1][0]])
print ('Max value is at index [{}][{}] = {}'.format(arrayOfIndexes[0][0],arrayOfIndexes[1][0],slopeNumpyArray[arrayOfIndexes[0][0]][arrayOfIndexes[1][0]]))
print ('Min value is at index [{}][{}] = {}'.format(arrayOfMinIndexes[0][0], arrayOfMinIndexes[1][0],slopeNumpyArray[arrayOfMinIndexes[0][0]][arrayOfMinIndexes[1][0]]))
print ('Average value is {}'.format(getAverage(slopeTifFile)))






# get origin and pixel size of fe tif file
print('LP_GRS_Fe_Global_2ppd.tif')
transform = feTiffFile.GetGeoTransform()
if transform:
    print("Origin of FE tiff file = ({}, {})".format(transform[0], transform[3]))
    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for fe tif file
indexOfMax = getMaxIndex2d(feTiffFile)
indexOfMin = getMinIndex2d(feTiffFile)
print("Max value of Fe Global tiff file is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], feNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
print('Min value of Fe Global tiff file is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], feNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
print('Mean value of Fe Global tiff file is {}'.format(getAverage(feTiffFile)))



# get origin and pixel size of lola dem tif file
print('LRO_LOLA_DEM_Global_128ppd_v04.tif')
transform = lolademFile.GetGeoTransform()   # overwrite our old variable transform
if transform:
    print("Origin of lola dem tiff file = ({}, {})".format(transform[0], transform[3]))
    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for lola dem tif file
indexOfMax = getMaxIndex2d(lolademFile)
indexOfMin = getMinIndex2d(lolademFile)
print("Max value of lola dem tiff file is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], lolaNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], lolaNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
print('Mean value of lola dem tiff file is {}'.format(getAverage(lolademFile)))
print("Projection is {}".format(lolademFile.GetProjection()))
x = getXCoordinate(indexOfMax[0][0])
y = getYCoordinate(indexOfMax[1][0])
print('max x coordinate : {}'.format(x))
print('max y coordinate : {}'.format(y))



# get origin and pixel size of H tif file
print('LP_GRS_H_Global_2ppd.tif')
transform = hTifFile.GetGeoTransform()   # overwrite our old variable transform
if transform:
    print("Origin of H tiff file = ({}, {})".format(transform[0], transform[3]))
    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for H tif file
indexOfMax = getMaxIndex2d(hTifFile)
indexOfMin = getMinIndex2d(hTifFile)
print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], hNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], hNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
print('Mean value of H tiff file is {}'.format(getAverage(hTifFile)))




# get origin and pixel size of K tif file
print('LP_GRS_K_Global_halfppd.tif')
transform = kTifFile.GetGeoTransform()   # overwrite our old variable transform
if transform:
    print("Origin of K tiff file = ({}, {})".format(transform[0], transform[3]))
    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for K tif file
indexOfMax = getMaxIndex2d(kTifFile)
indexOfMin = getMinIndex2d(kTifFile)
print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], kNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], kNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
print('Mean value is {}'.format(getAverage(kTifFile)))


# get origin and pixel size of Th tif file
print('LP_GRS_Th_Global_2ppd.tif')
transform = thTifFile.GetGeoTransform()   # overwrite our old variable transform
if transform:
    print("Origin of Th tif file = ({}, {})".format(transform[0], transform[3]))
    print("Pixel Size = ({}, {})".format(transform[1], transform[5]))

# calculate max, min, and average for Th tif file
indexOfMax = getMaxIndex2d(thTifFile)
indexOfMin = getMinIndex2d(thTifFile)
print("Max value is at index [{}][{}] = {}".format(indexOfMax[0][0], indexOfMax[1][0], thNumpyArray[indexOfMax[0][0]][indexOfMax[1][0]]))
print('Min value is at index [{}][{}] = {}'.format(indexOfMin[0][0], indexOfMin[1][0], thNumpyArray[indexOfMin[0][0]][indexOfMin[1][0]]))
print('Mean value is {}'.format(getAverage(thTifFile)))
