# -*- coding: utf-8 -*-
"""
@author: Radoslaw Guzinski
Copyright: (C) 2017, Radoslaw Guzinski
"""

import math
import os

import numpy as np
import scipy.ndimage as ndi

from osgeo import gdal


def getRasterInfo(raster):
    proj = raster.GetProjection()
    gt = raster.GetGeoTransform()
    sizeX = raster.RasterXSize
    sizeY = raster.RasterYSize
    extent = [gt[0], gt[3]+gt[5]*sizeY, gt[0]+gt[1]*sizeX, gt[3]]
    bands = raster.RasterCount
    return proj, gt, sizeX, sizeY, extent, bands


def resampleWithGdalWarp(srcFile, templateFile, outFile="", outFormat="MEM",
                         resampleAlg="average"):
    # Get template projection, extent and resolution
    try:
        proj, gt, sizeX, sizeY, extent, _ = getRasterInfo(templateFile)
    except AttributeError:
        templateDs = gdal.Open(templateFile)
        proj, gt, sizeX, sizeY, extent, _ = getRasterInfo(templateDs)
        templateDs = None

    # Resample with GDAL warp
    outDs = gdal.Warp(outFile,
                      srcFile,
                      format=outFormat,
                      dstSRS=proj,
                      xRes=gt[1],
                      yRes=gt[5],
                      outputBounds=extent,
                      resampleAlg=resampleAlg)

    return outDs


def point2pix(point, gt, upperBound=False):
    mx = point[0]
    my = point[1]
    if not upperBound:
        px = math.floor((mx - gt[0]) / gt[1])  # x pixel
        py = math.floor((my - gt[3]) / gt[5])  # y pixel
    else:
        px = math.ceil((mx - gt[0]) / gt[1])  # x pixel
        py = math.ceil((my - gt[3]) / gt[5])  # y pixel
    return [int(px), int(py)]


def pix2point(pix, gt):
    px = pix[0]
    py = pix[1]
    mx = px*gt[1] + gt[0]  # x coordinate
    my = py*gt[5] + gt[3]  # y coordinate
    return [mx, my]


# save the data to geotiff or memory
def saveImg(data, geotransform, proj, outPath, noDataValue=np.nan, fieldNames=[]):

    # Start the gdal driver for GeoTIFF
    if outPath == "MEM":
        driver = gdal.GetDriverByName("MEM")
        driverOpt = []
        is_netCDF = False
    else:
        # If the output file has .nc extension then save it as netCDF,
        # otherwise assume that the output should be a GeoTIFF
        ext = os.path.splitext(outPath)[1]
        if ext.lower() == ".nc":
            driver = gdal.GetDriverByName("netCDF")
            driverOpt = ["FORMAT=NC2"]
            is_netCDF = True
        else:
            driver = gdal.GetDriverByName("GTiff")
            driverOpt = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'BIGTIFF=IF_SAFER']
            is_netCDF = False

    shape = data.shape
    if len(shape) > 2:
        ds = driver.Create(outPath, shape[1], shape[0], shape[2], gdal.GDT_Float32, driverOpt)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        for i in range(shape[2]):
            ds.GetRasterBand(i+1).WriteArray(data[:, :, i])
            ds.GetRasterBand(i+1).SetNoDataValue(noDataValue)
    else:
        ds = driver.Create(outPath, shape[1], shape[0], 1, gdal.GDT_Float32, driverOpt)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        ds.GetRasterBand(1).WriteArray(data)
        ds.GetRasterBand(1).SetNoDataValue(noDataValue)

    # In case of netCDF format use netCDF4 module to assign proper names
    # to variables (GDAL can't do this). Also it seems that GDAL has
    # problems assigning projection to all the bands so fix that.
    if is_netCDF and fieldNames:
        from netCDF4 import Dataset
        ds = None
        ds = Dataset(outPath, 'a')
        grid_mapping = ds["Band1"].grid_mapping
        for i, field in enumerate(fieldNames):
            ds.renameVariable("Band"+str(i+1), field)
            ds[field].grid_mapping = grid_mapping
        ds.close()
        ds = gdal.Open('NETCDF:"'+outPath+'":'+fieldNames[0])

    print('Saved ' + outPath)

    return ds


def binomialSmoother(data):
    def filterFunction(footprint):
        weight = [1, 2, 1, 2, 4, 2, 1, 2, 1]
        # Don't smooth land and invalid pixels
        if np.isnan(footprint[4]):
            return footprint[4]

        footprintSum = 0
        weightSum = 0
        for i in range(len(weight)):
            # Don't use land and invalid pixels in smoothing of other pixels
            if not np.isnan(footprint[i]):
                footprintSum = footprintSum + weight[i] * footprint[i]
                weightSum = weightSum + weight[i]
        try:
            ans = footprintSum/weightSum
        except ZeroDivisionError:
            ans = footprint[4]
        return ans

    smoothedData = ndi.filters.generic_filter(data, filterFunction, 3)

    return smoothedData


def appendNpArray(array, data, axis=None):
    if array is None or array.size == 0:
        array = data
    else:
        array = np.append(array, data, axis=axis)
    return array


# Reproject and subset the given low resolution datasets to high resolution
# scene projection and extent
def reprojectSubsetLowResScene(highResScene, lowResScene, resampleAlg=gdal.GRA_Bilinear):

    # Read the required metadata
    xsize_HR = highResScene.RasterXSize
    ysize_HR = highResScene.RasterYSize
    gt_HR = highResScene.GetGeoTransform()
    proj_HR = highResScene.GetProjection()

    # Reproject low res scene to high res scene's projection to get the original
    # pixel size in the new projection
    out = gdal.Warp("",
                    lowResScene.GetDescription(),
                    format="MEM",
                    dstSRS=proj_HR,
                    resampleAlg=gdal.GRA_NearestNeighbour)

    # Make the new LR pixel as close as possible to original low resolution while
    # overlapping nicely with the high resolution pixels
    gt_LR = out.GetGeoTransform()
    pixSize_HR = [gt_HR[1], math.fabs(gt_HR[5])]
    pixSize_LR = [round(gt_LR[1]/pixSize_HR[0])*pixSize_HR[0],
                  round(math.fabs(gt_LR[5])/pixSize_HR[0])*pixSize_HR[0]]
    out = None

    # Make the extent such that it does not go outside high resolution extent so that the matrix
    # is the same size as resampled high resolution reflectances in the next step
    UL = [gt_HR[0], gt_HR[3]]
    xsize_LR = int((xsize_HR*pixSize_HR[0])/pixSize_LR[0])
    ysize_LR = int((ysize_HR*pixSize_HR[1])/pixSize_LR[1])
    BR = [UL[0] + xsize_LR*pixSize_LR[0], UL[1] - ysize_LR*pixSize_LR[1]]

    # Call GDAL warp to reproject and subsset low resolution scene
    out = gdal.Warp("",
                    lowResScene.GetDescription(),
                    format="MEM",
                    dstSRS=proj_HR,
                    xRes=pixSize_LR[0],
                    yRes=pixSize_LR[1],
                    outputBounds=[UL[0], BR[1], BR[0], UL[1]],
                    resampleAlg=resampleAlg)

    return out


# Resample high res scene to low res pixel while extracting homogeneity
# statistics. It is assumed that both scenes have the same projection and extent.
def resampleHighResToLowRes(highResScene, lowResScene):

    gt_HR = highResScene.GetGeoTransform()
    gt_LR = lowResScene.GetGeoTransform()
    xSize_LR = lowResScene.RasterXSize
    ySize_LR = lowResScene.RasterYSize

    aggregatedMean = np.zeros((ySize_LR,
                               xSize_LR,
                               highResScene.RasterCount))
    aggregatedStd = np.zeros(aggregatedMean.shape)

    # Calculate how many high res pixels are grouped in a low res pixel
    pixGroup = [int(gt_LR[5]/gt_HR[5]), int(gt_LR[1]/gt_HR[1])]

    # Go through all the high res bands and calculate mean and standard
    # deviation when aggregated to the low resolution
    for band in range(highResScene.RasterCount):
        data_HR = highResScene.GetRasterBand(band+1).ReadAsArray(0, 0, pixGroup[1]*xSize_LR,
                                                                 pixGroup[0]*ySize_LR)
        t = data_HR.reshape(ySize_LR, pixGroup[0], xSize_LR, pixGroup[1])
        t = t.transpose(0, 2, 1, 3).reshape(ySize_LR, xSize_LR, pixGroup[0]*pixGroup[1])
        aggregatedMean[:, :, band] = np.nanmean(t, axis=-1)
        aggregatedStd[:, :, band] = np.nanstd(t, axis=-1)

    return aggregatedMean, aggregatedStd
