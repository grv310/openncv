import cv2


facecascade  = cv2.CascadeClassifier(r'C:\Users\gaurav\Desktop\project\face_detection\haarcascade_frontalface_default.xml')


image= cv2.imread(r'C:\Users\gaurav\Desktop\project\face_detection\abba.png')


gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)



faces = facecascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=4)



print(type(faces))
 
print(faces)

for x,y,w,h in faces:
    img=cv2.rectangle(image,(x,y),(x+w,x+h),(0,0,255),3)
    
cv2.imshow("gray",image)


cv2.waitKey(2000)


cv2.destroyAllWindows()