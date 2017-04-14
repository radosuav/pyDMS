# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 17:16:37 2017

@author: radoslaw guzinski
"""
import os
import time

from osgeo import gdal

import pyDMSUtils as utils
from pyDMS import DecisionTreeSharpener, NeuralNetworkSharpener
from pyDMS import REG_sknn_ann, REG_sklearn_ann

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
    residualImage, correctedImage = disaggregator.residualAnalysis(downscaledFile, lowResFilename, lowResMaskFilename, doCorrection = True)
    print("Saving output...")
    highResFile = gdal.Open(highResFilename) 
    if correctedImage is not None: 
        outImage = correctedImage
    else:
        outImage = downscaledFile
    #outData = utils.binomialSmoother(outData)
    outFile = utils.saveImg(outImage.GetRasterBand(1).ReadAsArray(), 
                            outImage.GetGeoTransform(), 
                            outImage.GetProjection(), 
                            outputFilename)
    residualFile = utils.saveImg(residualImage.GetRasterBand(1).ReadAsArray(),
                                 residualImage.GetGeoTransform(),
                                 residualImage.GetProjection(),
                                 os.path.splitext(outputFilename)[0]+"_residual"+os.path.splitext(outputFilename)[1])
    
    outFile = None
    residualFile = None
    downsaceldFile = None
    highResFile = None
        
    print(time.time() - start_time, "seconds")
