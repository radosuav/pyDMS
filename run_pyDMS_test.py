# -*- coding: utf-8 -*-
"""
@author: radoslaw guzinski
Copyright: (C) 2017, Radoslaw Guzinski
"""
import os
import time

from osgeo import gdal

import pyDMS.pyDMSUtils as utils
from pyDMS.pyDMS import DecisionTreeSharpener, NeuralNetworkSharpener
from pyDMS.pyDMS import REG_sknn_ann, REG_sklearn_ann

highResFilename = r"C:\Users\pako\Downloads\s2reprojected30m-2.tif"
lowResFilename = r"C:\Users\pako\Downloads\S3A_20170517T093144_Sobrino_lst_nadir_1km__fromNdvi_1km.tif"
outputFilename = r"C:\Users\pako\output-rf-10.tif"

##########################################################################################

if __name__ == "__main__":

    useDecisionTree = True

    commonOpts = {"highResFiles":               [highResFilename],
                  "lowResFiles":                [lowResFilename],
                  "cvHomogeneityThreshold":     0,
                  "movingWindowSize":           0,
                  "disaggregatingTemperature":  True}
    dtOpts =     {"method": "rf",
                  "perLeafLinearRegression":    False,
                  "linearRegressionExtrapolationRatio": 0.25}
    ensembleOpts = {"n_jobs": 4,
                    "n_estimators": 10}
    sknnOpts =   {'hidden_layer_sizes':         (10,),
                  'activation':                 'tanh'}
    nnOpts =     {"regressionType":             REG_sklearn_ann,
                  "regressorOpt":               sknnOpts}

    start_time = time.time()

    if useDecisionTree:
        opts = commonOpts.copy()
        opts.update(dtOpts)
        disaggregator = DecisionTreeSharpener(**opts, ensembleOpt=ensembleOpts)
    else:
        opts = commonOpts.copy()
        opts.update(nnOpts)
        disaggregator = NeuralNetworkSharpener(**opts)

    print("Training regressor...")
    disaggregator.trainSharpener()
    print("Sharpening...")
    downscaledFile = disaggregator.applySharpener(highResFilename, lowResFilename)
    print("Residual analysis...")
    residualImage, correctedImage = disaggregator.residualAnalysis(downscaledFile, lowResFilename,
                                                                   doCorrection=True)
    print("Saving output...")
    highResFile = gdal.Open(highResFilename)
    if correctedImage is not None:
        outImage = correctedImage
    else:
        outImage = downscaledFile
    # outData = utils.binomialSmoother(outData)
    outFile = utils.saveImg(outImage.GetRasterBand(1).ReadAsArray(),
                            outImage.GetGeoTransform(),
                            outImage.GetProjection(),
                            outputFilename)
    residualFile = utils.saveImg(residualImage.GetRasterBand(1).ReadAsArray(),
                                 residualImage.GetGeoTransform(),
                                 residualImage.GetProjection(),
                                 os.path.splitext(outputFilename)[0] + "_residual" +
                                 os.path.splitext(outputFilename)[1])

    outFile = None
    residualFile = None
    downsaceldFile = None
    highResFile = None

    print(time.time() - start_time, "seconds")
