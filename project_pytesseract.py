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

          img = image
          copy_img = img.copy()
          img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          cv2.imwrite('image_result/gray.jpg',img2)
          blur = cv2.GaussianBlur(img2,(3,3),0)                #노이즈 제거를 위해 blur처리
          cv2.imwrite('image_result/blur.jpg',blur)

          adaptive = cv2.adaptiveThreshold(
               blur,
               maxValue=255.0,
               adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
               thresholdType=cv2.THRESH_BINARY_INV,
               blockSize=19,
               C=9
          )

          sobel = cv2.Sobel(blur, -1, 1, 0, delta=128);
          canny = cv2.Canny(blur, 100, 200)
          laplacian = cv2.Laplacian(blur, cv2.CV_8U, ksize=5);

          cv2.imwrite('image_result/edge_adaptive.jpg', adaptive)
          cv2.imwrite('image_result/edge_canny.jpg',canny)
          cv2.imwrite('image_result/edge_sobel.jpg', sobel)
          cv2.imwrite('image_result/edge_laplacian.jpg', laplacian)

          contours, _  = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          contours2, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          contours3, _ = cv2.findContours(sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          contours4, _ = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

          box1 = []
          f_count= 0
          select=0
          plate_width=0

          box2 = []
          f_count2 = 0
          select2 = 0
          plate_width2 = 0

          box3 = []
          f_count3 = 0
          select3 = 0
          plate_width3 = 0

          box4 = []
          f_count4 = 0
          select4 = 0
          plate_width4 = 0


# adaptive
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
          cv2.imwrite('image_result/contours_adaptive.jpg',img)


          # -------------------------------------------------------------------------------#


# canny
          for i in range(len(contours2)):
               cnt = contours2[i]
               area = cv2.contourArea(cnt)
               x, y, w, h = cv2.boundingRect(cnt)
               rect_area = w * h  # area size
               aspect_ratio = float(w) / h  # ratio = width/height

               if (aspect_ratio >= 0.2) and (aspect_ratio <= 1.0) and (rect_area >= 100) and (rect_area <= 700):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    box2.append(cv2.boundingRect(cnt))

          for i in range(len(box2)):  ##Buble Sort on python
               for j in range(len(box2) - (i + 1)):
                    if box2[j][0] > box2[j + 1][0]:
                         temp = box2[j]
                         box2[j] = box2[j + 1]
                         box2[j + 1] = temp

          # to find number plate measureing length between rectangles
          for m in range(len(box2)):
               count = 0
               for n in range(m + 1, (len(box2) - 1)):
                    delta_x = abs(box2[n + 1][0] - box2[m][0])
                    if delta_x > 150:
                         break
                    delta_y = abs(box2[n + 1][1] - box2[m][1])
                    if delta_x == 0:
                         delta_x = 1
                    if delta_y == 0:
                         delta_y = 1
                    gradient = float(delta_y) / float(delta_x)
                    if gradient < 0.25:
                         count = count + 1
               # measure number plate size
               if count > f_count2:
                    select2 = m
                    f_count2 = count;
                    plate_width = delta_x
          cv2.imwrite('image_result/contours_canny.jpg', img)


          # -------------------------------------------------------------------------------#


# sobel
          for i in range(len(contours3)):
               cnt = contours3[i]
               area = cv2.contourArea(cnt)
               x, y, w, h = cv2.boundingRect(cnt)
               rect_area = w * h  # area size
               aspect_ratio = float(w) / h  # ratio = width/height

               if (aspect_ratio >= 0.2) and (aspect_ratio <= 1.0) and (rect_area >= 100) and (rect_area <= 700):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    box3.append(cv2.boundingRect(cnt))

          for i in range(len(box3)):  ##Buble Sort on python
               for j in range(len(box3) - (i + 1)):
                    if box3[j][0] > box3[j + 1][0]:
                         temp = box3[j]
                         box3[j] = box3[j + 1]
                         box3[j + 1] = temp

          # to find number plate measureing length between rectangles
          for m in range(len(box3)):
               count = 0
               for n in range(m + 1, (len(box3) - 1)):
                    delta_x = abs(box3[n + 1][0] - box3[m][0])
                    if delta_x > 150:
                         break
                    delta_y = abs(box3[n + 1][1] - box3[m][1])
                    if delta_x == 0:
                         delta_x = 1
                    if delta_y == 0:
                         delta_y = 1
                    gradient = float(delta_y) / float(delta_x)
                    if gradient < 0.25:
                         count = count + 1
               # measure number plate size
               if count > f_count3:
                    select3 = m
                    f_count3 = count;
                    plate_width = delta_x
          cv2.imwrite('image_result/contours_sobel.jpg', img)


          # -------------------------------------------------------------------------------#


# laplacian
          for i in range(len(contours4)):
               cnt = contours4[i]
               area = cv2.contourArea(cnt)
               x, y, w, h = cv2.boundingRect(cnt)
               rect_area = w * h  # area size
               aspect_ratio = float(w) / h  # ratio = width/height

               if (aspect_ratio >= 0.2) and (aspect_ratio <= 1.0) and (rect_area >= 100) and (rect_area <= 700):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    box4.append(cv2.boundingRect(cnt))

          for i in range(len(box4)):  ##Buble Sort on python
               for j in range(len(box4) - (i + 1)):
                    if box4[j][0] > box4[j + 1][0]:
                         temp = box4[j]
                         box4[j] = box4[j + 1]
                         box4[j + 1] = temp

          # to find number plate measureing length between rectangles
          for m in range(len(box4)):
               count = 0
               for n in range(m + 1, (len(box4) - 1)):
                    delta_x = abs(box4[n + 1][0] - box4[m][0])
                    if delta_x > 150:
                         break
                    delta_y = abs(box4[n + 1][1] - box4[m][1])
                    if delta_x == 0:
                         delta_x = 1
                    if delta_y == 0:
                         delta_y = 1
                    gradient = float(delta_y) / float(delta_x)
                    if gradient < 0.25:
                         count = count + 1
               # measure number plate size
               if count > f_count4:
                    select4 = m
                    f_count4 = count;
                    plate_width = delta_x
          cv2.imwrite('image_result/contours_laplacian.jpg', img)


          # -------------------------------------------------------------------------------#


          number_plate=copy_img[box1[select][1]-10:box1[select][3]+box1[select][1]+20,box1[select][0]-10:140+box1[select][0]]
          resize_plate=cv2.resize(number_plate,None,fx=1.8,fy=1.8,interpolation=cv2.INTER_CUBIC+cv2.INTER_LINEAR)
          plate_gray=cv2.cvtColor(resize_plate,cv2.COLOR_BGR2GRAY)
          ret,th_plate = cv2.threshold(plate_gray,150,255,cv2.THRESH_BINARY)

          cv2.imwrite('image_result/plate_th.jpg',th_plate)
          kernel = np.ones((3,3),np.uint8)
          er_plate = cv2.erode(th_plate,kernel,iterations=1)             # erosion 밝은 영역을 줄임
          er_invplate = er_plate
          cv2.imwrite('image_result/erode_plate.jpg',er_invplate)
          result = pytesseract.image_to_string(Image.open('image_result/erode_plate.jpg'), config='--psm 6')
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

# playsound.playsound('50000.mp3')