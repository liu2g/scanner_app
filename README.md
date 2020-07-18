# Document Scanner Program
Crop and process document pictures for better reading

## Requirements
The software dependencies of the program is listed below.
```
Python 3.7.7
pathlib2
numpy V1.18.5
opencv V3.4.2
matplotlib V3.2.2
```
To install all required packages, run
```
pip install -r requirements.txt
```
**Note:** The exact version of the package is not required, though it is suggested to use the same software dependencies.

## Usage
The program is broken into steps, each of which is written as a function in `scanner_lib.py`.
`example.py` is a main controller program that processes all images in `../imgs` and outputs to `../results` with the same file name. More testing can be done by adding image files in `../imgs`.
To run the example program:
```
python example.py
```
Or
```
py example.py
```
Depending on the system.

## Future Work
The project is written under the assumption that no Machine Learning is involved. An amount of future work can be done to perfect the program for better performance and compatibility.