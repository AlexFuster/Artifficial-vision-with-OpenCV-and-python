# Artifficial-vision-with-OpenCV-and-python
We use the  OpenCV  library and the  Python  language in  order to develop different  guiding algorithms  on a mobile autonomous own-manufactured robot,  which includes a  camera connected to a  Raspberry PI.  The purpose of these algorithms  is to solve  five  challenges  through  the extraction of guiding parameters from  the analysis of the visual  feedback offered by the camera, as well as use these parameters to control the robot.

THIS CODE CAN'T BE EXECUTED OR TESTED WITHOUT THE PROPPER PHYSICAL INFRAESTRUCTURE: A ROBOT WITH A RASPBERRY PI 3 WITH CAMERA CONNECTED TO AN ARDUINO UNO THAT CONTROLS A SERVO AND TWO MOTORS. IN ADDITION, THE RESOURCES(IMAGES) REFFERENCED IN THE CODE ARE NOT INCLUDED IN THIS REPOSITORY.

The four code files in this repository are executable:

    -ClientSocket.py and ClientSocket_GUI.py can be executed in any device with python 2. They are tcp clients to connect with the server and send it commands. The only difference between them is that the second version uses a graphic interface.
    
    -ServerSocket4.py can only be executed as it is in a raspberry PI with camera. The code is specifically designed to this device, although it can be adapted to work on a pc. It receives commands from the client sockets while it uses artifficil vision techniques to     calculate routes based on the the camera captures. It also sends commands to an arduino microcontroller
    
    -prueba_serial.ino can only be executed in an arduino microcontroller. This code directly controls the servos and motors of the robot
