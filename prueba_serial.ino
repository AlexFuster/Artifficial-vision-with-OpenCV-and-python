
#include <Servo.h>

#define PIN_IN1 4
#define PIN_IN2 2
#define PIN_IN3 8
#define PIN_IN4 7
#define EN_A 6
#define EN_B 5
#define PIN_SERVO 11

const int MOT_L_MIN = -255;
const int MOT_L_MAX = 255;
const int MOT_R_MIN = -255;
const int MOT_R_MAX = 255;
const int VL = 80;
const int VR = 100;
const int VG = 140;
Servo servo;
String str;
int angle = 90;
String out[20];
int params[20];
String order;
int talla;
int i;
int sta;
int t0=0,t1=0;
bool parametros,escuchar;
int vl=0,vr=0;
void motores(int mot_L, int mot_R)
{
  //Restringe el rango de la entrada
  mot_L=constrain(mot_L,-255,255);
  mot_R=constrain(mot_R,-255,255);
  //Mapea la entrada a los rangos mÃ¡ximos de los motores para asegurar proporcionalidad


  mot_L=map(mot_L,-255,255,MOT_L_MIN,MOT_L_MAX);
  mot_R=map(mot_R,-255,255,MOT_R_MIN,MOT_R_MAX);
  //LÃ³gica de decisiÃ³n:
  if(mot_R>0)
    digitalWrite(PIN_IN1,HIGH);
  else{
    digitalWrite(PIN_IN1,LOW);
    if (mot_R<0)
      digitalWrite(PIN_IN2,HIGH);
    else
      digitalWrite(PIN_IN2,LOW);  
  }

  if(mot_L>0)
    digitalWrite(PIN_IN3,HIGH);
  else{
    digitalWrite(PIN_IN3,LOW);
    if (mot_L<0)
      digitalWrite(PIN_IN4,HIGH);
    else
      digitalWrite(PIN_IN4,LOW);  
  }

  analogWrite(EN_B,abs(mot_L));
  analogWrite(EN_A,abs(mot_R));
     
}

int split(String in, char sep){
  int talla = 0;
  int pos;
  
  while((pos=in.indexOf(sep)) != -1){
    out[talla++] = in.substring(0,pos);
    if(pos+1<= in.length()-1)
      in = in.substring(pos+1);
    else 
      in = "";
  }
  if(in.length()>0)
    out[talla++] = in;
  return talla;
}

void setV(int vi,int vd){
  vl=vi;
  vr=vd;
}
void reset(){
  setV(0,0);
}  

void setup()
{
  Serial.begin(57600);

  pinMode(PIN_IN1,OUTPUT);
  pinMode(PIN_IN2,OUTPUT);
  pinMode(PIN_IN3,OUTPUT);
  pinMode(PIN_IN4,OUTPUT);
  servo.attach(PIN_SERVO);
  
  servo.write(angle);
}
 
void loop()
{
  if (Serial.available()) { //Si está disponible
     str = Serial.readStringUntil(';'); //Guardamos la lectura en una variable char
     talla = split(str,':');
     order = out[0];
     //Serial.println(order);
     parametros = false;
     if(talla>1){
       parametros = true;
       talla = split(out[1],',');
       for(i=0;i<talla;i++){
        params[i] = out[i].toInt();
        //Serial.println(params[i]);
       }
     }
     t1=millis();
     //Serial.println(t1-t0);
     t0=t1;
     if(order == "escuchar"){
       escuchar=true;
     }
     else if(order == "reset"){
       reset();
       angle = 90;
       servo.write(angle);
     }
     else if(escuchar){
       if(order == "recto"){
          if(parametros)
            setV(params[0],params[1]);
          else
            setV(VL,VR);
          sta=0;
       }
       else if(order == "atras"){
          if(parametros)
            setV(-params[0],-params[1]);
          else
            setV(-VL,-VR);
       }
       else if(order == "para"){
          setV(0,0);
          sta=3;
       }
       else if(order == "der"){
          if(parametros)
            setV(params[0],0);
          else
            setV(VG,0);
       }
       else if(order == "izq"){
          if(parametros)
            setV(0,params[0]);
          else
            setV(0,VG);
       }
       else if(order == "der2"){
          if(parametros)
            setV(params[0],-params[0]);
          else
            setV(VG,-VG);
       }
       else if(order == "izq2"){
          if(parametros)
            setV(-params[0],params[0]);
          else
            setV(-VG,VG);
       }
       else if(order == "inclinar"){
          if(parametros)
            angle = min(180,angle+params[0]);
          else
            angle = min(150,angle+30);
          servo.write(angle);
       }
       else if(order == "reclinar"){
          if(parametros)
            angle = max(30,angle-params[0]);
          else
            angle = max(30,angle-30);
          servo.write(angle);
       }
       else if(order == "servo"){
          if(parametros && params[0]>=0 && params[0]<=180){
            angle = params[0];
            servo.write(angle);
          }
       }
       else if(order == "p"){
        setV(0,0);
        escuchar=false;
        sta=3;
       }
       else if(order == "exit"){
        setV(0,0);
        angle=90;
        servo.write(angle);
        escuchar=false;
       }
       else if(order == "l"){
         if(sta==2||sta==3)
           reset();
         sta=1;
         setV(max(0,vl-30),min(255,vr+30));
       }
       else if(order == "r"){
         if(sta==1||sta==3)
           reset();
         sta=2;
         setV(min(255,vl+30),max(0,vl-30));
       }
       motores(vl,vr);
     }   
  }
}

  

