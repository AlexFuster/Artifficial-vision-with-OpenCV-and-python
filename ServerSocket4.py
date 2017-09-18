#!/usr/bin/env python
import socket
import select
from multiprocessing import Process, Queue, Pool
from functools import partial
import sys
import numpy as np, cv2
import time
import pickle
import serial
import copy as cp
from picamera.array import PiRGBArray
from picamera import PiCamera


def servidor(q,arduinoactivo,arduino): 
    read_list = []
    serversocket    =   socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    serversocket.bind(('', 8000))
    serversocket.listen(1)
    while 1:     
        if len(read_list) == 0:
            clientsocket, address = serversocket.accept()
            read_list = [clientsocket]
            print ('Connection from %s' %str(address))
        
        readable, writable, errored = select.select(read_list, [], [])
        
        if len(readable) >0:
            
            s = readable[0]
            data = s.recv(128) 
            if data:
                if arduinoactivo: 
                    arduino.write(data)
                data = data.decode('utf-8')
                s.send(('He Recibido '+ data).encode('utf-8'))
                print ('cliente %s' %data) 
                q.put(data)

            else:
                s.close()
                read_list.remove(s)
    
def findContours(img,mode): #detecta objetos blancos sobre fondo negro y escoge el mas grande 
    img2 = cp.copy(img)
    _, contours ,_ = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>=1:
        if len(contours)>1:
            contours= sorted(contours,key=lambda tup: cv2.contourArea(tup), reverse = True)
        cnt = contours[0]
        
        if mode==1:
            if len(contours)<50:
                rect = cv2.boundingRect(cnt)
                x,y,w,h=rect
                cx = x+w/2    
            else:
                return None,None,0
        elif mode==2:
            rect = cv2.minAreaRect(cnt)
            cx = int(rect[0][0])
        return rect,cnt,cx-CX
    else:
        return None,None,0

def controlMotores(difx,minV=140,maxV=180): #control proporcional
    minD=20.0
    maxD=300.0
    secondary=0                
    if abs(difx)<=minD:
        arduino.write('recto:;'.encode('utf-8'))
    else:
        
        if abs(difx)<=80:
            secondary=80
        
        if abs(difx)<=80:
            minV=120
            maxV=160
        
        v=minV+int((maxV-minV)*(abs(difx)-minD)/(maxD-minD))
        print(difx,difdifx,v,secondary)
        if difx>0:
            arduino.write(('recto:'+str(v)+','+str(secondary)+';').encode('utf-8'))
        else:
            arduino.write(('recto:'+str(secondary)+','+str(v)+';').encode('utf-8'))

def controlMotores2(difx,difdifx,minV=120,maxDd=320,incv1=40,incv2=95): #control proporcional derivativo
    minD=20.0
    maxD=320.0
    secondary=0
    difx=min(difx,maxD)
    difdifx=min(difdifx,maxDd)
    if abs(difx)<=minD:
        arduino.write('recto:;'.encode('utf-8'))
    else:
        if abs(difx)<=100:
            secondary=80 
        v= int(minV + (abs(difx)-minD)*incv1/(maxD-minD) + max(difdifx,0)*incv2/maxDd)
        if difx>0:
            arduino.write(('recto:'+str(v)+','+str(secondary)+';').encode('utf-8'))
        else:
            arduino.write(('recto:'+str(secondary)+','+str(v)+';').encode('utf-8'))            
        

def visionm4(img1,img2,hacerfoto=False): #vision de la prueba 4
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = brisk.detectAndCompute(img1,None)
        kp2, des2 = brisk.detectAndCompute(img2,None)
        try:
            matches = bf2.knnMatch(des1,des2, k=2)

            good = []
            pts = []
            suma=0
            
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
                    pts.append(kp2[m.trainIdx].pt)
            
            pts = np.array(pts,np.float32)
            print(pts.shape[0])            
        except:
            pts=None      
            
        img1 = cv2.drawKeypoints(img1,kp1,None,color=(0,255,0),flags=0)
        img2 = cv2.drawKeypoints(img2,kp2,None,color=(0,255,0),flags=0)
        img2= cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
        cv2.namedWindow('Frame')
        cv2.moveWindow('Frame',0,0)
        cv2.imshow('Frame', img2)
        key = cv2.waitKey(1000)
        if hacerfoto:
            takephoto(img2,'')
        cv2.destroyAllWindows()
        return pts
        

def controlm4(tsleep,sentido='izq',tgiro='2',vrect=150,vgiro=200): #control de la prueba 4
        if sentido == 'recto':
            arduino.write('recto:'+str(vrect)+','+str(vrect)+';'.encode('utf-8'))
        else:
            arduino.write(sentido+tgiro+':'+str(vgiro)+';'.encode('utf-8'))
        time.sleep(tsleep)
        arduino.write('para:;'.encode('utf-8'))
        time.sleep(0.5)
           
def takephoto(img,nombrefoto): #la camara toma captura
    if len(nombrefoto) > 0:
        cv2.imwrite('fotos/'+nombrefoto+'.png',img)
    else:
        global photocount
        cv2.imwrite('fotos/captura_'+str(photocount)+'.png',img)
        photocount += 1
        f = open('photocount','w')
        f.write(str(photocount))
        f.close()

def callbackm3(event,X,Y,flags,param): #funcion evento para la inicializacion de la prueba 3
    if event == cv2.EVENT_LBUTTONDOWN:
        global countm3,x,y,w,h
        print(X,Y)
        if countm3==0:
            x=X
            y=Y
        elif countm3==1:
            w=X-x
        else:
            h=Y-y
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.destroyAllWindows()
            cv2.namedWindow('preparacion modo3')
            cv2.moveWindow('preparacion modo3',0,0)
            cv2.imshow('preparacion modo3',img)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        
        countm3+=1

def ajustarAnguloCamara(minval,maxval,cy): #ajuste de la altura de la camara
    global angle
    if cy > 0.75*rows and angle<maxval:
        angle+=10
        arduino.write('inclinar:10;'.encode('utf-8'))
    elif cy < 0.25*rows and angle>minval:
        angle-=10
        arduino.write('reclinar:10;'.encode('utf-8'))

def secuenciam5(orders): #control para la prueba 5
    for order,tsleep in orders:
        arduino.write('para:;'.encode('utf-8'))
        time.sleep(0.5)
        arduino.write(order.encode('utf-8'))
        time.sleep(tsleep)
    arduino.write('para:;'.encode('utf-8'))

def deteccionm5(img,template): #vision de la prueba 5
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    return cv2.minMaxLoc(res)[1]
if __name__ == '__main__':
    arduinoactivo = False
    photocount =0   
    videocount = 0
    try:
        f = open('photocount','r')
        photocount = int(f.read())
        
    except:
        f = open('photocount','w')
        f.write('0')
    finally:
        f.close()
        
    try:
        f = open('videocount','r')
        videocount = int(f.read())
        
    except:
        f = open('videocount','w')
        f.write('0')
    finally:
        f.close()    
    
    try:
        arduino = serial.Serial('/dev/ttyACM0', 57600)
        arduinoactivo = True
        
    except:
        print('Error en la comunicacion con el arduino')
    mode=0
    
    priority_tasks=[] 
    qorders = Queue() 
    servidorprocess = Process(target=servidor, args=(qorders,arduinoactivo,arduino,))
    servidorprocess.start() #creacion del proceso servidor 
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    tsleepm4=0.27
    tsleepm5=0.5
    nfotosm4=5
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    time.sleep(0.1)
    if arduinoactivo:
        arduino.write('escuchar:;'.encode('utf-8'))
    
    lower_green = np.array([50,150,50])
    upper_green = np.array([70,255,255])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    recordvideo=False
    angle=90
    brisk = cv2.BRISK_create()
    bf2 = cv2.BFMatcher(cv2.NORM_HAMMING)
    mostrarcaptura=False
    t0=0
    olddifx = 1000
    kernel = np.ones((5,5),np.uint8)
    pool=Pool(4)
    
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True): #bucle principal
        img = frame.array
        rows,cols,_ = img.shape
        CX = cols/2
        rect=None
        if recordvideo:
            out1.write(img)

        #modos de ejecucion 
        if mode==1:#linea
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)                    
            rect,pts,difx = findContours(img,1)
            if rect is not None:
                if np.sign(difx)!=np.sign(olddifx):
                    difdifx=0
                else:
                    difdifx = abs(difx)-abs(olddifx)              
                if arduinoactivo:    
                    controlMotores2(difx,difdifx,120,160,40,75)
 
                olddifx = difx

        elif mode==2:#pelota
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.inRange(img, lower_green, upper_green)
            img = cv2.erode(img,kernel,iterations=1)
            rect,pts,difx = findContours(img,2)
            difdifx = abs(difx)-np.sign(difx)*np.sign(olddifx)*abs(olddifx)
            olddifx = difx
            if rect is not None:
                cx,cy = int(rect[0][0]),int(rect[0][1])
                if arduinoactivo:
                    ajustarAnguloCamara(60,120,cy)
                    controlMotores2(difx,difdifx)

            
            
        elif mode==3:#matrioshka
            if countm3==0: #introduccion de la ventana inicial
                cv2.namedWindow('preparacion modo3')
                cv2.moveWindow('preparacion modo3',0,0)
                cv2.setMouseCallback('preparacion modo3',callbackm3)
                cv2.imshow('preparacion modo3',img)
                cv2.waitKey(0)
                
                track_window = (x,y,w,h)
                print(track_window)
                roi = img[y:y+h, x:x+w]
                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                roi_hist = cv2.calcHist([hsv_roi],[0,1],mask,[180,256],[0,180,0,256])
                cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                countm3+=1
                
            
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array((0., 60.,64.)), np.array((180.,255.,255.)))
                dst = cv2.calcBackProject([hsv],[0,1],roi_hist,[0,180,0,256],1)
                dst=cv2.bitwise_and(dst,dst,mask=mask)
                                
                dst = cv2.erode(dst,kernel,iterations=1)                
                
                _,maxval,_,maxpos=cv2.minMaxLoc(dst)
                _,maxval2,_,_=cv2.minMaxLoc(dst[y:y+h, x:x+w])
                print(maxval)
                if maxval>200:
                    if maxval > 2*maxval2:
                        x,y=maxpos
                        track_window=x,y,w,h
                    rect, track_window = cv2.CamShift(dst, track_window, term_crit)
                    cx,cy = int(rect[0][0]),int(rect[0][1])
                    difx=cx-CX
                    difdifx = abs(difx)-np.sign(difx)*np.sign(olddifx)*abs(olddifx)
                    olddifx = difx
                    x,y,w,h = track_window
                    
                    if arduinoactivo:
                        ajustarAnguloCamara(90,120,cy)
                        controlMotores2(difx,difdifx)
                
            
                
        elif mode==4:#libros
            
            if countm4 == 1: #busqueda de la caratura
                print('Encontrado')
                pts=visionm4(imgsource,img)
                if pts is not None:
                    rect = cv2.minAreaRect(pts)
                    difx = rect[0][0]-CX
                    if abs(difx)<20:
                        countm4+=1
                        controlm4(2,'recto')
                    
                    else:
                        if np.sign(difx)!= np.sign(olddifx):
                            tsleepm4_1 /= 2
                        if difx<0:
                            controlm4(tsleepm4_1)
                        else:
                            controlm4(tsleepm4_1,'der')
                    olddifx = difx
                              
            else:    
                pts= visionm4(imgsource,img)
                if pts is not None:
                    aux=pts.shape[0]
    	            if aux > 50:
                        countm4+=1
                        tsleepm4_1=tsleepm4        
                    else:
                        controlm4(tsleepm4)
           
        
        elif mode==5:#señales
            img = cv2.resize(img,(200,150))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            if countm5==0: #inicializacion
                arduino.write('recto:100,140;'.encode('utf-8'))
                countm5 +=1

            elif countm5==1:
                score = []
                
                funcm5 = partial(deteccionm5,img)
                listaux=pool.map(funcm5,templates)
                score=[(listaux[i],i) for i in range(len(listaux))]
    	        score = sorted(score,reverse=True)
    	        print(score)
                
                if score[0][0]>0.5:
    	            i = score[0][1]
    	            print(i)
                    if arduinoactivo:
                        izqder = ['izq2:;','der2:;']
                        countm5 =0
                        if i==0:
                            posizqder = int(2*np.random.random())
                            orders=[(izqder[posizqder],tsleepm5),('recto:100,140;',1.5),(izqder[1-posizqder],tsleepm5),('recto:100,140;',0.75)]
                            secuenciam5(orders)
                            orders.reverse()
                            secuenciam5(orders)
                        elif i==1 or i==2:
                            orders=[(izqder[i-1],0.7)]
                            secuenciam5(orders)
                        elif i==3:
                            arduino.write('para:;'.encode('utf-8'))
                            countm5 =2    
                                            


                 
        if recordvideo:
            out2.write(img)

        #realimentacion
        if mostrarcaptura:
            if rect is not None:
                if mode==1 or mode==2:
                    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                elif mode ==3 or mode ==4:
                    pts = np.int0(cv2.boxPoints(rect))                
                if mode==1:
                    x,y,w,h=rect
                    cx=x+w/2

                cv2.circle(img,(cx,rows),10,(255,0,0),2)
                cv2.circle(img,(CX,rows),10,(0,0,255),2)
                img = cv2.polylines(img,[pts],True, (0,255,0),2)
            cv2.namedWindow('Frame')
            cv2.moveWindow('Frame',0,0)
            cv2.imshow('Frame', img)
            key = cv2.waitKey(1)
        
        #ejecucion de ordenes
        while not qorders.empty(): 
            new_task = qorders.get() 
            priority_tasks += [new_task] 
        if priority_tasks != []: 
            tarea = priority_tasks.pop() 
            print('Tarea escogida: %s' %tarea)
            params = tarea.split(':')[1][:-1]
            if 'exit' in tarea: 
                servidorprocess.terminate()
                if arduinoactivo:
                    arduino.close()
                f.close()
                sys.exit()
            elif 'foto' in tarea:
                takephoto(img,params)
            elif 'video' in tarea:
                if recordvideo:
                    out1.release()
                    out2.release()
                    
                else:
                    if len(params) > 0:
                        nomvideo= 'videos/sin_procesar_'+params+'.avi'
                        nomvideo2= 'videos/procesado_'+params+'.avi'
                    else:
                        nomvideo= 'videos/sin_procesar_'+str(videocount)+'.avi'
                        nomvideo2= 'videos/procesado_'+str(videocount)+'.avi'
                    
                    videocount += 1
                    f = open('videocount','w')
                    f.write(str(videocount))
                    f.close()

                    out1 = cv2.VideoWriter(nomvideo,fourcc, 32.0, (640,480))
                    out2 = cv2.VideoWriter(nomvideo2,fourcc,32.0,(640,480))
                recordvideo= not recordvideo
                
            elif 'mode' in tarea:
                camera.contrast=0
                camera.saturation=0

                if len(params) > 0:
                    mode = int(params)
                    if arduinoactivo:
                        if mode==1:
                            angle=180
                        elif mode==2:
                            camera.contrast=100
                            camera.saturation=50
                            angle=90
                        elif mode==3:
                            countm3=0
                            
                            camera.contrast=100
                            camera.saturation=50
                            term_crit = ( cv2.TERM_CRITERIA_EPS, 100, 0.1 )
                                
                        elif mode==4:
                            angle=80
                            izqder = ['izq','der']
                            controlm4(0.5+2*np.random.random(),izqder[int(2*np.random.random())])
                            imgsource = cv2.imread('libros/libro'+str(np.random.random_integers(nfotosm4))+'.png',0)
                            countm4=0

                        elif mode==5:
                            angle=80                            
                            templates = [cv2.imread('patterns/cono2.png',0),cv2.imread('patterns/izq.png',0),cv2.imread('patterns/der.png',0),cv2.imread('patterns/stop.png',0)]
                            coloresm5 = [(0,255,255),(0,255,0),(255,0,0),(0,0,255)]
                            countm5=0

                                              
                else:
                    mode=0
                    if arduinoactivo:
                        angle=90
                arduino.write('servo:'+str(angle)+';'.encode('utf-8'))
                time.sleep(2)
            elif 'reset' in tarea:
                mode=0
                camera.contrast=0
                camera.saturation=0
            elif 'realimentacion' in tarea:
                mostrarcaptura = not mostrarcaptura
        
        
        rawCapture.truncate(0)
            
