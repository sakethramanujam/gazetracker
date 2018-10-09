import cv2
import autopy
from matplotlib import pyplot as plt
import seaborn as sns

ESCAPE_KEY = 27


face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_righteye_2splits.xml'
)

#number signifies camera
capture = cv2.VideoCapture(0)
x = list()
y = list()

while 1:
    success, image = capture.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray)

   for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(
            img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
        )
        roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
        roi_color2 = image[ey:ey + eh, ex:ex + ew]
       circles = cv2.HoughCircles(
            roi_gray2,
            cv2.HOUGH_GRADIENT,
            1,
            200,
            param1=200,
            param2=1,
            minRadius=0,
            maxRadius=0
        )
        print((ex+ew)/2,(ey+eh)/2)
        x.append((ex+ew)/2)
        y.append((ey+eh)/2)
        try:
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(
                    roi_color2,
                    (i[0], i[1]),
                    i[2],
                    (255, 255, 255),
                    2
                )
                # print("drawing circle")
                # draw the center of the circle
                cv2.circle(
                    roi_color2,
                    (i[0], i[1]),
                    2,
                    (255, 255, 255),
                    3
                )

                # print(i[0],i[1])
                
                x_pos = int((ex + ew) / 2)
                y_pos = int((ey + eh) / 2)
                autopy.mouse.move(x_pos, y_pos)
        except Exception as e:
            print('Exception:', e)

    cv2.imshow('img', image)
    key_pressed = cv2.waitKey(30) & 0xff
    if key_pressed == ESCAPE_KEY:
        break


capture.release()
cv2.destroyAllWindows()
data = list(zip(x, y))

print(data)

plt.scatter(x, y)
plt.show()
sns.set()
# plt.imshow(data, cmap='hot', interpolation='nearest')
