# -*- coding: utf-8 -*-
"""
@author: Radoslaw Guzinski
Copyright: (C) 2017, Radoslaw Guzinski
"""

import math
import os

import numpy as np
import scipy.ndimage as ndi
from numba import njit, stencil

from osgeo import gdal
from pyproj import Proj, Transformer


def openRaster(raster):
    closeOnExit = False
    try:
        raster.GetProjection()
        openRaster = raster
    except AttributeError:
        openRaster = gdal.Open(str(raster))
        closeOnExit = True
    return openRaster, closeOnExit


def getRasterInfo(raster):
    r, closeOnExit = openRaster(raster)
    proj = r.GetProjection()
    gt = r.GetGeoTransform()
    sizeX = r.RasterXSize
    sizeY = r.RasterYSize
    extent = [gt[0], gt[3]+gt[5]*sizeY, gt[0]+gt[1]*sizeX, gt[3]]
    bands = r.RasterCount
    if closeOnExit:
        r = None
    return proj, gt, sizeX, sizeY, extent, bands


def resampleWithGdalWarp(srcFile, templateFile, outFile="", outFormat="MEM",
                         resampleAlg="average", multithread=True):
    # Get template projection, extent and resolution
    proj, gt, sizeX, sizeY, extent, _ = getRasterInfo(templateFile)

    # Resample with GDAL warp
    outDs = gdal.Warp(str(outFile),
                      openRaster(srcFile)[0],
                      format=outFormat,
                      dstSRS=proj,
                      xRes=gt[1],
                      yRes=gt[5],
                      outputBounds=extent,
                      resampleAlg=resampleAlg,
                      multithread=multithread,
                      warpOptions={"NUM_THREADS": "ALL_CPUS"})

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
def saveImg(data, geotransform, proj, outPath, noDataValue=None, fieldNames=[]):
    outPath = str(outPath)

    # Save to memory first
    is_netCDF = False
    memDriver = gdal.GetDriverByName("MEM")
    shape = data.shape
    if len(shape) > 2:
        ds = memDriver.Create("MEM", shape[1], shape[0], shape[2], gdal.GDT_Float32)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        for i in range(shape[2]):
            ds.GetRasterBand(i+1).WriteArray(data[:, :, i])
    else:
        ds = memDriver.Create("MEM", shape[1], shape[0], 1, gdal.GDT_Float32)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        ds.GetRasterBand(1).WriteArray(data)

    # Save to file if required
    if outPath == "MEM":
        if noDataValue is None:
            noDataValue = np.nan
        ds.GetRasterBand(1).SetNoDataValue(noDataValue)
    else:
        # If the output file has .nc extension then save it as netCDF,
        # otherwise assume that the output should be a GeoTIFF (COG)
        ext = os.path.splitext(outPath)[1]
        if ext.lower() == ".nc":
            fileFormat = "netCDF"
            driverOpt = ["FORMAT=NC2"]
            is_netCDF = True
        else:
            fileFormat = "COG"
            driverOpt = ['COMPRESS=DEFLATE', 'PREDICTOR=YES', 'BIGTIFF=IF_SAFER']
        out_ds = gdal.Translate(outPath, ds, format=fileFormat, creationOptions=driverOpt,
                                noData=noDataValue, stats=True)
        # If GDAL driers for other formats do not exist then default to GeoTiff
        if out_ds is None:
            print("Warning: Selected GDAL driver is not supported! Saving as GeoTiff!")
            fileFormat = "GTiff"
            driverOpt = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'BIGTIFF=IF_SAFER']
            is_netCDF = False
            ds = gdal.Translate(outPath, ds, format=fileFormat, creationOptions=driverOpt,
                                noData=noDataValue, stats=True)
        else:
            ds = out_ds

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
    proj_HR, gt_HR, xsize_HR, ysize_HR, extent = getRasterInfo(highResScene)[0:5]
    proj_LR, gt_LR, xsize_LR, ysize_LR = getRasterInfo(lowResScene)[0:4]

    # Transform "middle pixel" and "middle pixel + 1" between LR and HR projections
    # to obtain LR resolution in HR projection. This method can handle different
    # x and y resolution
    midPix = [int(xsize_LR/2), int(ysize_LR/2)]
    midPix_2 = [midPix[0]+1, midPix[1]+1]
    midPoint = pix2point(midPix, gt_LR)
    midPoint_2 = pix2point(midPix_2, gt_LR)
    transformer = Transformer.from_proj(Proj(proj_LR), Proj(proj_HR), always_xy=True)
    x1, y1 = transformer.transform(midPoint[0], midPoint[1])
    x2, y2 = transformer.transform(midPoint_2[0], midPoint_2[1])
    xRes_LR_proj = x2 - x1
    yRes_LR_proj = y2 - y1
    # Now do the same with UL pixel to obtain low resolution geotransform in
    # the new projection
    UL_x, UL_y = transformer.transform(gt_LR[0], gt_LR[3])
    gt_LR = [UL_x, xRes_LR_proj, 0, UL_y, 0, yRes_LR_proj]

    # Now subset to high resolution scene extent while not shifting pixels
    UL = pix2point(point2pix([extent[0], extent[3]], gt_LR, upperBound=False), gt_LR)
    BR = pix2point(point2pix([extent[2], extent[1]], gt_LR, upperBound=True), gt_LR)
    out = gdal.Warp("",
                    openRaster(lowResScene)[0],
                    format="MEM",
                    dstSRS=proj_HR,
                    resampleAlg=gdal.GRA_NearestNeighbour,
                    xRes=gt_LR[1],
                    yRes=gt_LR[5],
                    outputBounds=[UL[0], BR[1], BR[0], UL[1]])

    return out


# Resample high res scene to low res pixel while extracting homogeneity
# statistics. It is assumed that both scenes have the same projection and extent.
def resampleHighResToLowRes(highResScene, lowResScene):

    gt_HR = getRasterInfo(highResScene)[1]
    bands_HR = getRasterInfo(highResScene)[5]
    gt_LR, xSize_LR, ySize_LR = getRasterInfo(lowResScene)[1:4]
    xRes_HR = gt_HR[1]
    yRes_HR = abs(gt_HR[5])
    xRes_LR = gt_LR[1]
    yRes_LR = gt_LR[5]

    aggregatedMean = np.zeros((ySize_LR,
                               xSize_LR,
                               bands_HR))
    aggregatedStd = np.zeros(aggregatedMean.shape)

    # Go through all the high res bands and calculate mean and standard
    # deviation when aggregated to the low resolution
    highRes, close = openRaster(highResScene)
    for band in range(bands_HR):
        bandData_HR = highRes.GetRasterBand(band+1).ReadAsArray().astype(np.float32)
        nodataValue = highRes.GetRasterBand(1).GetNoDataValue()
        bandData_HR[bandData_HR == nodataValue] = np.nan
        aggregatedMean[:, :, band], aggregatedStd[:, :, band] =\
            _resampleHighResToLowRes(bandData_HR, ySize_LR, yRes_LR, yRes_HR, xSize_LR, xRes_LR,
                                     xRes_HR, gt_HR, gt_LR)
    if close:
        highRes = None
    return aggregatedMean, aggregatedStd


@njit
def _resampleHighResToLowRes(bandData_HR, ySize_LR, yRes_LR, yRes_HR, xSize_LR, xRes_LR, xRes_HR,
                             gt_HR, gt_LR):
    aggregatedMean = np.zeros((ySize_LR, xSize_LR))
    aggregatedStd = np.zeros((ySize_LR, xSize_LR))
    for yPix_LR in range(ySize_LR):
        yPos_LR_min = gt_LR[3] + yPix_LR*yRes_LR
        yPix_HR_min = int(round(max(0, gt_HR[3] - yPos_LR_min) / yRes_HR))
        yPix_HR_max = int(round(max(0, gt_HR[3] - (yPos_LR_min + yRes_LR)) / yRes_HR))
        for xPix_LR in range(xSize_LR):
            xPos_LR_min = gt_LR[0] + xPix_LR*xRes_LR
            xPix_HR_min = int(round(max(0, xPos_LR_min - gt_HR[0]) / xRes_HR))
            xPix_HR_max = int(round(max(0, xPos_LR_min + xRes_LR - gt_HR[0]) / xRes_HR))
            aggregatedMean[yPix_LR, xPix_LR] =\
                np.nanmean(bandData_HR[yPix_HR_min:yPix_HR_max, xPix_HR_min:xPix_HR_max])
            aggregatedStd[yPix_LR, xPix_LR] =\
                np.nanstd(bandData_HR[yPix_HR_min:yPix_HR_max, xPix_HR_min:xPix_HR_max])

    return aggregatedMean, aggregatedStd


def resampleLowResToHighRes(lowResScene, highResScene, resampleAlg="cubic"):

    lowResScene_resampled = resampleWithGdalWarp(lowResScene, highResScene,
                                                 resampleAlg=resampleAlg)

    data = lowResScene_resampled.GetRasterBand(1).ReadAsArray()
    # Sometimes there can be 1 HR pixel NaN border arond LR invalid pixels due to resampling.
    # Fuction below fixes this. Image border pixels are excluded due to numba stencil
    # limitations.
    data[1:-1, 1:-1] = removeEdgeNaNs(data)[1:-1, 1:-1]

    # The resampled array might be slightly smaller then the high-res because
    # of the subsetting of the low resolution scene. In that case just pad
    # the missing values with neighbours.
    highResShape = (getRasterInfo(highResScene)[3], getRasterInfo(highResScene)[2])
    if highResShape != data.shape:
        data = np.pad(data,
                      ((0, highResShape[0]-data.shape[0]), (0, highResShape[1]-data.shape[1])),
                      "edge")
    return data


@stencil(cval=1.0)
def removeEdgeNaNs(a):
    if np.isnan(a[0, 0]) and (not np.isnan(a[-1, 0]) or not np.isnan(a[1, 0]) or
                              not np.isnan(a[0, -1]) or not np.isnan(a[0, 1])):
        return np.nanmean(np.array([a[-1, 0], a[1, 0], a[0, -1], a[0, 1]]))
    else:
        return a[0, 0]
