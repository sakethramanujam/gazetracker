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
# Standards: use constants like this to follow DRY
# and reduce the occurrence of "magic numbers" in your code that lack context
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def update_mouse_position(hough_circles, eye_x_pos, eye_y_pos, roi_color2):
    try:
        for circle in hough_circles[0, :]:
            # Standards: DRY (don't repeat yourself), define circle_center once and use it twice.
            circle_center = (circle[0], circle[1])
            # draw the outer circle
            cv2.circle(
                img=roi_color2,
                center=circle_center,
                radius=circle[2],
                color=WHITE,
                thickness=2
            )
            # print("drawing circle")
            # draw the center of the circle
            cv2.circle(
                img=roi_color2,
                center=circle_center,
                radius=2,
                color=WHITE,
                thickness=3
            )

            # print(i[0],i[1])

            x_pos = int(eye_x_pos)
            y_pos = int(eye_y_pos)
            autopy.mouse.move(x_pos, y_pos)
    except Exception as e:
        # Standards: exception handling in try: except cases should generally be as specific as possible. What type of
        # exception are you expecting to encounter here? Instead of capturing "Exception" capture that specific class of exception.
        print('Exception:', e)

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_righteye_2splits.xml'
)

def run_tracker():
    #number signifies camera
    video_capture = cv2.VideoCapture(0)
    eye_x_positions = list()
    eye_y_positions = list()

    key_pressed = None

    while not key_pressed == ESCAPE_KEY:
        success, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = eye_cascade.detectMultiScale(gray)

        for (eye_x, eye_y, eye_width, eye_height) in eyes:
            # Standards: may be good to explicitly call out the parameters being passed to methods,
            # if you are not explicitly declaring these values as variables with informative names
            cv2.rectangle(
                img=image,
                pt1=(eye_x, eye_y),
                pt2=(eye_x + eye_width, eye_y + eye_height),
                color=GREEN,
                thickness=2
            )
            roi_gray2 = gray[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width]
            roi_color2 = image[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width]
            # Standards: As above, it may be good to describe the parameters being passed
            # but in this case I had a hard time reconciling the positional arguments here with expected parameters
            hough_circles = cv2.HoughCircles(
                roi_gray2,
                cv2.HOUGH_GRADIENT,
                1,
                200,
                param1=200,
                param2=1,
                minRadius=0,
                maxRadius=0
            )
            # Standards: DRY (don't repeat yourself), calculate eye positions once and use them below
            eye_x_pos = (eye_x + eye_width) / 2
            eye_y_pos = (eye_y + eye_height) / 2
            print(eye_x_pos, eye_y_pos)
            eye_x_positions.append(eye_x_pos)
            eye_y_positions.append(eye_y_pos)

            # Standards: in general in the interests of improving readability, move logical chunks
            # of code out into their own classes, methods, or functions to make it easier to understand overall
            # program flow
            update_mouse_position(hough_circles, eye_x_pos, eye_y_pos, roi_color2)

        cv2.imshow('img', image)

        # Standards: code like this can be hard to understand, so comments explicitly describing operation are desirable:
        # This reduces cv2.waitKey() response to 8 bit integer, representing ASCII input
        key_pressed = cv2.waitKey(30) & 0xff

    # close window
    video_capture.release()
    cv2.destroyAllWindows()

    # plot heatmap
    plot_data(eye_x_positions, eye_y_positions)

def plot_data(x, y):
    h, x, y, p = plt.hist2d(x, y)               # generates 2d heatmap
    plt.clf()
    plt.imshow(h, interpolation = "gaussian")   # draws heatmap
    plt.axis("off")
    plt.show()
    sns.set()

if __name__ == "__main__":
    run_tracker()
