from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import struct
import numpy
import sys

gtif = gdal.Open( "LRO_NAC_Slope_15m_20N010E_2mp.tif", gdal.GA_ReadOnly )
#numpy array should have no data value

cols = gtif.RasterXSize
rows = gtif.RasterYSize
bands = gtif.RasterCount

band = gtif.GetRasterBand(1)
print band
bandtype = gdal.GetDataTypeName(band.DataType)
print bandtype


'''
# print gtif[0][0]
band = gtif.GetRasterBand(1)
print "Raster 1 x size  {0}".format(gtif.GetRasterBand(1).XSize)
print "Raster 1 y size {0}".format(gtif.GetRasterBand(1).YSize)

bandtype = gdal.GetDataTypeName(band.DataType)
print bandtype
data = band.ReadAsArray(0, 0, gtif.RasterXSize, gtif.RasterYSize)

#print gtif
#print gtif.GetMetadata()
#print data

# this allows GDAL to throw Python Exceptions
print("Driver: {}/{}".format(gtif.GetDriver().ShortName,
                             gtif.GetDriver().LongName))
print("Size is {} x {} x {}".format(gtif.RasterXSize,
                                    gtif.RasterYSize,
                                    gtif.RasterCount))
print("Projection is {}".format(gtif.GetProjection()))
geotransform = gtif.GetGeoTransform()

print len(geotransform)
print geotransform[0]
print geotransform[1]
print geotransform[2]
print geotransform[3]
print geotransform[4]
print geotransform[5]


if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))


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
