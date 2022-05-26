import cv2
import numpy as np
import playsound as playsound
import pytesseract
from  PIL import Image

class Recognition:
     def ExtractNumber(self):
          capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
          capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
          capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
          ret, image = capture.read()
          # Number='testimg3.jpg'
          # img=cv2.imread(Number,cv2.IMREAD_COLOR)
          # copy_img=img.copy()
          img = image
          copy_img = img.copy()
          img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          cv2.imwrite('image_result/gray.jpg',img2)
          blur = cv2.GaussianBlur(img2,(3,3),0)
          cv2.imwrite('image_result/blur.jpg',blur)

          adaptive = cv2.adaptiveThreshold(
               blur,
               maxValue=255.0,
               adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
               thresholdType=cv2.THRESH_BINARY_INV,
               blockSize=19,
               C=9
          )
          cv2.imwrite('image_result/adaptive.jpg', adaptive)

          canny=cv2.Canny(blur,100,200)
          cv2.imwrite('image_result/canny.jpg',canny)

          contours, _  = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

          box1=[]
          f_count=0
          select=0
          plate_width=0

          for i in range(len(contours)):
               cnt=contours[i]
               area = cv2.contourArea(cnt)
               x,y,w,h = cv2.boundingRect(cnt)
               rect_area=w*h  #area size
               aspect_ratio = float(w)/h # ratio = width/height

               if  (aspect_ratio>=0.2)and(aspect_ratio<=1.0)and(rect_area>=100)and(rect_area<=700):
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                    box1.append(cv2.boundingRect(cnt))

          for i in range(len(box1)): ##Buble Sort on python
               for j in range(len(box1)-(i+1)):
                    if box1[j][0]>box1[j+1][0]:
                         temp=box1[j]
                         box1[j]=box1[j+1]
                         box1[j+1]=temp

         #to find number plate measureing length between rectangles
          for m in range(len(box1)):
               count=0
               for n in range(m+1,(len(box1)-1)):
                    delta_x=abs(box1[n+1][0]-box1[m][0])
                    if delta_x > 150:
                         break
                    delta_y =abs(box1[n+1][1]-box1[m][1])
                    if delta_x ==0:
                         delta_x=1
                    if delta_y ==0:
                         delta_y=1
                    gradient =float(delta_y) /float(delta_x)
                    if gradient<0.25:
                        count=count+1
               #measure number plate size
               if count > f_count:
                    select = m
                    f_count = count;
                    plate_width=delta_x
          cv2.imwrite('image_result/snake.jpg',img)


          number_plate=copy_img[box1[select][1]-10:box1[select][3]+box1[select][1]+20,box1[select][0]-10:140+box1[select][0]]
          resize_plate=cv2.resize(number_plate,None,fx=1.8,fy=1.8,interpolation=cv2.INTER_CUBIC+cv2.INTER_LINEAR)
          plate_gray=cv2.cvtColor(resize_plate,cv2.COLOR_BGR2GRAY)
          ret,th_plate = cv2.threshold(plate_gray,150,255,cv2.THRESH_BINARY)

          cv2.imwrite('image_result/plate_th.jpg',th_plate)
          kernel = np.ones((3,3),np.uint8)
          er_plate = cv2.erode(th_plate,kernel,iterations=1)
          er_invplate = er_plate
          cv2.imwrite('image_result/er_plate.jpg',er_invplate)
          result = pytesseract.image_to_string(Image.open('image_result/er_plate.jpg'), config='--psm 6')
          return(result.replace(" ",""))

recogtest=Recognition()
result=recogtest.ExtractNumber()
num = ''
for i in result:
     if ('0' <= i) and (i <= '9'):
          num += i
print(num)

if(num == '1000'):
     playsound.playsound('1000.mp3')
elif(num == '5000'):
     playsound.playsound('5000.mp3')
elif(num == '10000'):
     playsound.playsound('10000.mp3')
elif(num == '50000'):
     playsound.playsound('50000.mp3')

