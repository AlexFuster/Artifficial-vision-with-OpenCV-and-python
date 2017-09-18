from Tkinter import *
import socket
import sys

localip = 'localhost'
raspip = '192.168.1.8'
#raspip = '192.168.2.4'
ip = raspip

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)# creamos el socket
clientsocket.connect((ip,8000))# ahora hacemos que se conecte con el servidor

top = Tk()

def callBack(msg):
    print(msg)
    msg+=';'
    clientsocket.send(msg.encode('utf-8')) 
    newdata = clientsocket.recv(1024).decode('utf-8')
    if 'exit' in msg: #en este caso, salimos
        clientsocket.close()
        sys.exit() 

fwd=PhotoImage(file="icons/fwd.gif")
bwd=PhotoImage(file="icons/bwd.gif")
stop=PhotoImage(file="icons/stop.gif")
horario=PhotoImage(file="icons/horario.gif")
antihorario=PhotoImage(file="icons/antihorario.gif")
reclinar=PhotoImage(file="icons/reclinar.gif")
inclinar=PhotoImage(file="icons/inclinar.gif")
increase=PhotoImage(file="icons/increase.gif")
photo=PhotoImage(file="icons/photo.gif")
power=PhotoImage(file="icons/cancel.gif")
escucha=PhotoImage(file="icons/escucha.gif")
noescucha=PhotoImage(file="icons/noescucha.gif")
reset=PhotoImage(file="icons/power.gif")
modo1=PhotoImage(file="icons/mode1.gif")
modo2=PhotoImage(file="icons/mode2.gif")
modo3=PhotoImage(file="icons/mode3.gif")
modo4=PhotoImage(file="icons/mode4.gif")
video=PhotoImage(file="icons/video.gif")
mostrarcaptura=PhotoImage(file="icons/mostrar.gif")
modo5=PhotoImage(file="icons/mode5.gif")

Button(top, image =antihorario, command =lambda:callBack('izq2:')).grid()
Button(top, image =fwd, command =lambda: callBack('recto:')).grid(row=0,column=1)
Button(top, image =horario, command =lambda: callBack('der2:')).grid(row=0,column=2)

Button(top, image =fwd, command =lambda: callBack('izq:')).grid(row=1)
Button(top, image =fwd, command =lambda: callBack('der:')).grid(row=1,column=2)
Button(top, image =stop, command =lambda: callBack('para:')).grid(row=1,column=1)

Button(top, image =increase, command =lambda: callBack('l:')).grid(row=2)
Button(top, image =increase, command =lambda: callBack('r:')).grid(row=2,column=2)
Button(top, image =bwd, command =lambda: callBack('atras:')).grid(row=2,column=1)

Button(top, image =reclinar, command =lambda: callBack('reclinar:')).grid(row=0,column=3)
Button(top, image =inclinar, command =lambda: callBack('inclinar:')).grid(row=1,column=3)
Button(top, image =photo, command =lambda: callBack('foto:')).grid(row=2,column=3)

Button(top, image =escucha, command =lambda: callBack('escuchar:')).grid(row=3)
Button(top, image =power, command =lambda: callBack('exit:')).grid(row=3,column=1)
Button(top, image =noescucha, command =lambda: callBack('p:')).grid(row=3,column=2)

Button(top, image =reset, command =lambda: callBack('reset:')).grid(row=3,column=3)

Button(top, image =modo1, command =lambda: callBack('mode:1')).grid(row=0,column=4)
Button(top, image =modo2, command =lambda: callBack('mode:2')).grid(row=1,column=4)
Button(top, image =modo3, command =lambda: callBack('mode:3')).grid(row=2,column=4)
Button(top, image =modo4, command =lambda: callBack('mode:4')).grid(row=3,column=4)
Button(top, image =video, command =lambda: callBack('video:')).grid(row=4)
Button(top, image =mostrarcaptura, command =lambda: callBack('realimentacion:')).grid(row=4,column=1)
Button(top, image =modo5, command =lambda: callBack('mode:5')).grid(row=4,column=4)


top.mainloop()
