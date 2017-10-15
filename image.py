from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import struct
import numpy
import sys

#function that returns index of max value
def getMaxIndex2d(openedGDALfile):
    rasterBand = gtif.GetRasterBand(1)
    noValue = rasterBand.GetNoDataValue()
    tiffArray = numpy.array(openedGDALfile.ReadAsArray())
    return numpy.where((tiffArray != noValue) & (tiffArray == numpy.max(tiffArray)))


gtif = gdal.Open( "LRO_NAC_Slope_15m_20N010E_2mp.tif", gdal.GA_ReadOnly )
#numpy array should have no data value

cols = gtif.RasterXSize
rows = gtif.RasterYSize
bands = gtif.RasterCount


band = gtif.GetRasterBand(1)
print type(band) # <class 'osgeo.gdal.Band'>
bandtype = gdal.GetDataTypeName(band.DataType)
print bandtype # Float32

# shows no data value
print "NO DATA VALUE :", band.GetNoDataValue() # -1.0
noDataValue = band.GetNoDataValue()

# converts file opened to a numpy array
tiffArray = numpy.array(gtif.ReadAsArray())
print type(tiffArray) # <type 'numpy.ndarray'>
print tiffArray.ndim # 2 = the number of axes (dimensions) of the array. In the Python world, the number of dimensions is referred to as rank.
print tiffArray.size # 77677770  = 14695*5286 the total number of elements of the array. This is equal to the product of the elements of shape.
print tiffArray.shape # (14695, 5286) = (rows, columns)
#print tiffArray # remove comment to print the array
#print tiffArray[0][0]  # top left
#print tiffArray[14694][5285]  # bottom right

noDataValueCounter = 0
sum = 0
arrayOfIndexes=getMaxIndex2d(gtif)
print 'Max value is at index [', arrayOfIndexes[0][0], '][', arrayOfIndexes[1][0], '] = ', tiffArray[arrayOfIndexes[0][0]][arrayOfIndexes[1][0]]

# second way of finding max values index
print numpy.argwhere((tiffArray == numpy.max(tiffArray)) & (tiffArray != noDataValue))

feTiffFile = gdal.Open("LP_GRS_Fe_Global_2ppd.tif",gdal.GA_ReadOnly)
tarray = numpy.array(feTiffFile.ReadAsArray())
indexOfMax = getMaxIndex2d(feTiffFile)

geotransform = gtif.GetGeoTransform()
if geotransform:
    print("Origin of slope tiff = ({}, {})".format(geotransform[0], geotransform[3]))

transform = feTiffFile.GetGeoTransform()
if transform:
    print("Origin of FE tiff file = ({}, {})".format(transform[0], transform[3]))

print 'Max value of Fe Global tiff file is at index [', indexOfMax[0][0], '][', indexOfMax[1][0], '] = ', tarray[indexOfMax[0][0]][indexOfMax[1][0]]
''''
geotransform = gtif.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

# if you do gdal info on the file opened
# you see that size is 5286, 14695
# i goes from 0 - 14694
# j goes from 0 - 5285
for i in range(len(tiffArray)):
    for j in range(len(tiffArray[0])):
        #print 'tiffArray[',i,'][',j, '] = value ',tiffArray[i][j]
        if tiffArray[i][j] == noDataValue:
            # need to not use # in computation
            noDataValueCounter += 1
        else:
            sum += tiffArray[i][j]
            print 'SUM: ', sum

average = sum/(tiffArray.size-noDataValueCounter)
print 'Average = ', average

'''

'''
# this allows GDAL to throw Python Exceptions
print("Driver: {}/{}".format(gtif.GetDriver().ShortName,
                             gtif.GetDriver().LongName))
print("Size is {} x {} x {}".format(gtif.RasterXSize,
                                    gtif.RasterYSize,
                                    gtif.RasterCount))
print("Projection is {}".format(gtif.GetProjection()))
geotransform = gtif.GetGeoTransform()



# bands data
band = gtif.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min, max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min, max))

if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

#for i in range(rows):
scanline = band.ReadRaster(xoff=0, yoff=0,
                           xsize=band.XSize, ysize=1,
                           buf_xsize=band.XSize, buf_ysize=1,
                           buf_type=gdal.GDT_Float32)
#print scanline
#print type(scanline)
tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
print tuple_of_floats
print type(tuple_of_floats)
print tuple_of_floats.__sizeof__()
print rows*cols
'''
