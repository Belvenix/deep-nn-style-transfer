# NST-Gradient


## Installation
1. Make sure you have Docker on your machine
2. Run the following command to build an image:
```$cmd
docker build . -t "dnn"
```
3. Run the following command to run the image as container:
```$cmd
docker run -dit --name dnn -v $PROJECT_ROOT/deep-nn-style-transfer:/repo dnn
```
__NOTE__: You should manually change $PROJECT_ROOT variable
4. Enter docker:
```cmd
docker exec -it dnn bash
```
5. ???
6. Profit

## Other
To stop the container use the following command (or gui):
```
docker stop dnn
```

To start the container use the following command (or gui):
```
docker start dnn
```

To check your containers use the following command (or gui):
```
docker ps -al
```

Copyright disclaimer: We do NOT own style images featured in this repository. All rights belong to it's rightful owner/owner's. No copyright infringement intended. 