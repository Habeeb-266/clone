 import cv2
 import numpy as np
 import time

    # Start the webcam
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    # Capture background
    background = None
    for i in range(50):
        ret, background = cap.read()

    background = cv2.flip(background, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV range for **PINK color**
        lower_pink = np.array([140, 50, 80])
        upper_pink = np.array([180, 255, 255])

        # Create mask for pink color
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Refine mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        mask_inv = cv2.bitwise_not(mask)

        # Apply cloak effect
        res1 = cv2.bitwise_and(background, background, mask=mask)
        res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

        cv2.imshow("Invisible Cloak - Pink", final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
