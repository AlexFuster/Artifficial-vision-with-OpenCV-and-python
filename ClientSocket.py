#!/usr/bin/env python
import socket
import sys
import numpy as np, cv2


localip = 'localhost'
raspip = '192.168.1.5'
ip = raspip

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)# creamos el socket
clientsocket.connect((ip,8000))# ahora hacemos que se conecte con el servidor
while 1:
    data = raw_input() #f recibimos datos del usuario
    if not data: #si no hay datos, terminamos
      break
    if ':' in data:#si el formato introducido es el correcto
        data += ';'
        clientsocket.send(data.encode('utf-8')) # enviamos los dtaos al servidor
        newdata = clientsocket.recv(1024).decode('utf-8') # recibimos acknowledge del servidor
        print ('servidor: %s' % newdata) 
        if 'exit' in data: #en este caso, salimos
            clientsocket.close()
            sys.exit()
    else:
        print('Orden no valida')
clientsocket.close()