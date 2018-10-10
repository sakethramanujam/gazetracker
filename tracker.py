from sys import platform as sys_pf
if sys_pf == 'darwin':
    # necessary for matplotlib on OSX, see https://stackoverflow.com/questions/43066073/matplotlib-tkinter-opencv-crashing-in-python-3
    import matplotlib
    matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
import cv2
import autopy
import seaborn as sns

ESCAPE_KEY = 27
face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_righteye_2splits.xml'
)

# main function for running gaze tracker
def run_tracker():
    key_pressed = None

    #number signifies camera
    capture = cv2.VideoCapture(0)
    x = list()
    y = list()

    while not key_pressed == ESCAPE_KEY:
        success, image = capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = eye_cascade.detectMultiScale(gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
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
            #print((ex+ew)/2,(ey+eh)/2)
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

                    # draw the center of the circle
                    cv2.circle(
                        roi_color2,
                        (i[0], i[1]),
                        2,
                        (255, 255, 255),
                        3
                    )

                    x_pos = int((ex + ew) / 2)
                    y_pos = int((ey + eh) / 2)
                    autopy.mouse.move(x_pos, y_pos)
            except Exception as e:
                print('Exception:', e)

        cv2.imshow('img', image)
        key_pressed = cv2.waitKey(30) & 0xff

    capture.release()
    cv2.destroyAllWindows()

    plot_data(x, y)

    return x, y

def plot_data(x, y):
    h, x, y, p = plt.hist2d(x, y)               # generates 2d heatmap
    plt.clf()
    plt.imshow(h, interpolation = "gaussian")   # draws heatmap
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    x, y = run_tracker()
