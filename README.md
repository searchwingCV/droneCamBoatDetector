
# DEPRECATED CODE

For more info visit 

wiki: https://wiki.searchwing.org/en/home/Development/Payload-Camera/searchwing-bilderkennung



# ComputerVision for searchwing drones
![alt text](https://raw.githubusercontent.com/searchwingCV/droneCamBoatDetector/master/screenshot.jpg)
* Left: 3D View 
    * "Cam": 3D-Position and orientation of drone camera
    * Pink points: the visible area in the image
    * Coordinates: possible tracked boats with their ID
* Right: 2D View 
    * Yellow boxes: Regions of Interest(ROIs) in the current image which will be associated to existing trackings
    * Red boxes: ROIs in the image which are too big
    * Green crosses: Tracked positions (with ID) which got recognized over several frames and therefore considered as boats 
    * Blue crosses: Tracked positions which were created in previous frames
## Install dependencies
### Python 2.7  
Install Opencv3 for python 2.7:
```
python -m pip install opencv-python
```
### ROS
ROS handles the communication between the different modules, transforms 3d-data and provides a handy 3d visualization for development
* [Short introduction to ROS](https://courses.cs.washington.edu/courses/cse466/11au/calendar/ros_cc_1_intro-jrsedit.pdf)
* Install a ros-desktop-version for your specific OS : http://wiki.ros.org/melodic/Installation

## Build/integrate src to ros framework
* create a catkin_workspace to compile the ros-specific code: http://wiki.ros.org/catkin/Tutorials/create_a_workspace
* git clone this repo  
```
cd /catkin_ws/src/
https://github.com/searchwingCV/droneCamBoatDetector.git
```
* Compile/install  
``` 
cd /catkin_ws  
catkin_make install
source /catkin_ws/devel/setup.bash
```
* Download the bodenSee2018HalfRes.bag dataset:
https://cloud.hs-augsburg.de/index.php/s/fgXNLN2Qza6ercr

## Run on the bodensee-dataset
* New Terminal: Start roscore
```
roscore
```

* New Terminal: Start dataset playbag.
```
cd datasetDownloadPath
rosbag play bodenSee2018HalfRes.bag --clock --start=0 --rate=2
```

* New Terminal: 
Setup environment:
```
source /catkin_ws/devel/setup.bash
```
Start boatdetector:
```
roslaunch searchwing bodenseeDataset.launch
```
Now you should see the detections in rviz like in the screenshot above.


If you want to edit/start/debug the boadDetectorNode.py with a editor, please ensure you run
```
rosparam set use_sim_time true
```
before you start the editor in the terminal. Also comment out the start of the node in the bodenseeDataset.launch file.

    



