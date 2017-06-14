# Count Cellula: Automated cell quantification in python 
 
 
 
 
  
## Overview

Count cellula is designed to perform automated image segmentation across a z-stack of images.  To run this program, clone the reposity and run the `count-cellula.ipynb` notebook in the notebooks/ directory. A sample image is provided in data/raw/ .
  
  Analysis workflow:  
<ol>
<li> Blur the image or perform histogram analysis</li>
<li> Perform histogram equlization (if blurred in first step, otherwise blur)</li>
<li> Perform blob detection</li>
<li> Perform clustering analysis</li>
<li> .</li>


## Project organization

``` │
├── data/               <- The original, immutable data dump. 
│
├── figures/            <- Figures saved by scripts or notebooks.
│
├── notebooks/          <- Jupyter notebooks. Naming convention is a short `-` delimited 
│                         description, a number (for ordering), and the creator's initials,
│                        e.g. `initial-data-exploration-01-hg`.
│
├── output/					<- Manipulated data, logs, etc.
│
├── tests/					<- Unit tests.
│
├── src/		<- Python module with source code of this project.
│
├── environment.yml			<- conda virtual environment definition file.
│
├── LICENSE
│
├── Makefile            <- Makefile with commands like `make environment`
│
├── README.md           <- The top-level README for developers using this project.
│
└── tox.ini             <- tox file with settings for running tox; see tox.testrun.org
```
 This script is currently in ongoing development. 
   
   
 
   
   
   
   
 Cookie cutter style project organization. The original can be found [here](https://github.com/hgrif/example-project).
