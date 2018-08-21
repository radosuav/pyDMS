# pyDMS
Python implementation of Data Mining Sharpener (DMS): a decision tree based algorithm for 
sharpening (disaggregation) of low-resolution images using high-resolution images. 
The implementation is mostly based on [Gao2012].

The DMS is trained with high-resolution data resampled to
    low resolution and low-resolution data and then applied 
    directly to high-resolution data to obtain high-resolution representation
    of the low-resolution data.
    
The implementation includes selecting training data based on homogeneity
    statistics and using the homogeneity as weight factor ([Gao2012], section 2.2),
    performing linear regression with samples located within each regression
    tree leaf node ([Gao2012], section 2.1), using an ensemble of regression trees 
    ([Gao2012], section 2.1), performing local (moving window) and global regressions and
    combining them based on residuals ([Gao2012] section 2.3) and performing residual 
    analysis and bias correction ([Gao2012], section 2.4)
    
Additionally, the Decision Tree regressor can be replaced by Neural Network regressor.

To install, download the project to your local system, enter the download directory and then type

`python setup.py install` 

For usage template see [run_pyDMS.py](/run_pyDMS.py).

Copyright: (C) 2017, Radoslaw Guzinski

## References

* [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data 
       Mining Approach for Sharpening Thermal Satellite Imagery over Land. 
       Remote Sensing, 4(11), 3287–3319. https://doi.org/10.3390/rs4113287
       
## License

pyDMS: a Python Data Mining Sharpener implementation

Copyright 2018 Radoslaw Guzinski and contributors.
    
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
