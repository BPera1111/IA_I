import cv2 as cv


img = cv.imread("/home/lautaro/programacion/final_IA/algo/mechromancer build.jpg",0)
#Redimensionamos la imagen para que tenga un tamaño de 256x256
img = cv.resize(img, (256, 256))
#Convertimos la imagen a escala de grises
#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window
