# FIND
This is the repo hosting FIND automatic quality inspection package

### ---> Documentation is under construction! <---

## Data

20 images in png format are included in "/input_images/test_set_01" dir. Image filenames go from 01.png to 20.png.


## Usage

The application has been developed and tested on Linux Ubuntu 18.04.4 LTS 64-bit OS with Python 3.6.9.
Before correctly running the application you need to:
- Create a virtual environment containing the "/requirements.txt" versions of used libraries;
- Edit the IPCAM_PATH and CALIBRATION_CAM_PATH in "config.py" accordingly to the path of your local copy of FIND master repo.
```
  IPCAM_PATH = '<local path to cloned repo>/input_images/test_set_01/'
  CALIBRATION_CAM_PATH = '<local path to cloned repo>/input_images/camera_calibration/'
```
You can process an image by using this command from FIND dir:
```
  python inspect_mold.py -i 04.png
```
To load a random image from the test set you can use the --random argument:
```
  python inspect_mold.py -r
```
To visualize all the intermediate steps of processing you can add the --step argument in both single image and random image mode, as follows:
```
  python inspect_mold.py -i 04.png -s
 
  python inspect_mold.py -r -s
 ```
To execute the next step you need to press a key. By typing "q" key you will quit the app.

 ## Output

Each time an image is processed an output is generated in "/output" dir consisting of:
1. The original image after pre-processing step;
2. A JSON file with information about inspected mold. In particular, the automatic analysis can assign three different outcome to the current mold and generate the following warning messages:<br>
>  - Mold is ok<br>
>  - Warning: Mold may need to be inspected!<br>
>  - Warning: Mold needs to be inspected now!<br>
  
