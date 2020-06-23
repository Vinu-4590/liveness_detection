import cv2

def head_movement():
    try:
        # create cv2 window and initialize VC object
        cv2.namedWindow("Face Recognizer")
        vc = cv2.VideoCapture(0)

        # define necessary variables
        # fixed_x1, fixed_y1, fixed_x2, fixed_y2 = 100, 150, 250, 300
        prev_x1, prev_y1, prev_x2, prev_y2 = 0, 0, 0, 0
        frame_counter = 0
        COUNTER = 0

        #this variable is responsible for number of frames taken
        #please configure this variable based on latency
        frame_counter_max = 20

        # initialize haarscade frontal face detector
        font = cv2.FONT_HERSHEY_SIMPLEX
        haar_detector = cv2.CascadeClassifier('face_landmark_dat/haarcascade_frontalface_default.xml')

        # looping through the frames in live stream
        while vc.isOpened():

            # read the frame by frame
            _, frame = vc.read()
            img = frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar_detector.detectMultiScale(gray, 1.3, 5)

            # loop through all the faces detected
            for (x, y, w, h) in faces:
                curr_x1 = x
                curr_y1 = y
                curr_x2 = x + w
                curr_y2 = y + h

            if frame_counter != 0:
                delta = abs(curr_x1 - prev_x1) + abs(curr_y1 - prev_y1) + abs(curr_x2 - prev_x2) + abs(curr_y2 - prev_y2)
                if delta > 50:
                    COUNTER += 1
                print(delta)
                print(prev_x1)
                print(curr_x1)

            key = cv2.waitKey(100)

            # show the image
            cv2.imshow("Face Recognizer", img)

            # condition to exit vc2 capture
            if key == ord('q') or frame_counter > frame_counter_max:  # exit on q
                break

            frame_counter += 1
            prev_x1 = curr_x1
            prev_y1 = curr_y1
            prev_x2 = curr_x2
            prev_y2 = curr_y2

        # a bit of clean-up
        vc.release()
        cv2.destroyAllWindows()
        print(COUNTER)
        if COUNTER >= 3:
            return True
        else:
            return False

    except Exception as ex:
        return ex
