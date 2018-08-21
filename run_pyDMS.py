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

highResFilename = r""
lowResFilename = r""
lowResMaskFilename = r""
outputFilename = r""

##########################################################################################

if __name__ == "__main__":

    useDecisionTree = True

    commonOpts = {"highResFiles":               [highResFilename],
                  "lowResFiles":                [lowResFilename],
                  "lowResQualityFiles":         [lowResMaskFilename],
                  "lowResGoodQualityFlags":     [255],
                  "cvHomogeneityThreshold":     0,
                  "movingWindowSize":           15,
                  "disaggregatingTemperature":  True}
    dtOpts =     {"perLeafLinearRegression":    True,
                  "linearRegressionExtrapolationRatio": 0.25}
    sknnOpts =   {'hidden_layer_sizes':         (10,),
                  'activation':                 'tanh'}
    nnOpts =     {"regressionType":             REG_sklearn_ann,
                  "regressorOpt":               sknnOpts}

    start_time = time.time()

    if useDecisionTree:
        opts = commonOpts.copy()
        opts.update(dtOpts)
        disaggregator = DecisionTreeSharpener(**opts)
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
                                                                   lowResMaskFilename,
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
