import cv2.cv2 as cv
import numpy as np
import pockets as P
from kalmanfilter import KalmanFilter


def create_pocs():
    cam = cv.VideoCapture("sorting_1.mp4")

    ret, frame = cam.read()

    Frame_width = frame.shape[1]
    Frame_height = frame.shape[0]
    cimg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_blur = cv.medianBlur(cimg, 11)
    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=100, maxRadius=300)
    detected_circles = np.around(circles)
    d_c = np.array(detected_circles[0].astype(int))


    if len(d_c) > 5:
        raise ValueError("Too many pockets detected")

    s_i = np.argmax(d_c[:, 2])
    s_poc = P.Pocket("Source", (d_c[s_i, 0], d_c[s_i, 1]), d_c[s_i, 2])
    for i in range(5):
        if i == s_i:
            continue
        elif d_c[i, 0] < s_poc.center[0] and d_c[i, 1] < s_poc.center[1]:
            g_poc = P.Pocket("Green", (d_c[i, 0], d_c[i, 1]), d_c[i, 2])
        elif d_c[i, 0] < s_poc.center[0] and d_c[i, 1] > s_poc.center[1]:
            r_poc = P.Pocket("Red", (d_c[i, 0], d_c[i, 1]), d_c[i, 2])
        elif d_c[i, 0] > s_poc.center[0] and d_c[i, 1] < s_poc.center[1]:
            y_poc = P.Pocket("Yellow", (d_c[i, 0], d_c[i, 1]), d_c[i, 2])
        else:
            b_poc = P.Pocket("Blue", (d_c[i, 0], d_c[i, 1]), d_c[i, 2])

    cam.release()

    return s_poc, g_poc, y_poc, r_poc, b_poc


def obj_poc_convergence(pocks, obj_cen):
    for i in range(len(pocks)):
        val = np.square(pocks[i].radius) - np.square((obj_cen[0] - pocks[i].center[0])) - np.square(
            (obj_cen[1] - pocks[i].center[1]))
        if val > 0:
            return i
    return 5


def detect_color(frame, param):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    avg_H = 0
    count = 0
    for i in range(param[0], param[0] + param[2]):
        for j in range(param[1], param[1] + param[3]):
            avg_H += hsv[j, i, 0]
            # hsv[j, i, :] = 0
            count += 1
    avg_H = avg_H / count
    return avg_H


def color_mask(frame, ilowH, ihighH):
    ilowS = 80
    ihighS = 255
    ilowV = 50
    ihighV = 255
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv.inRange(hsv, lower_hsv, higher_hsv)
    return mask


def main(pok):
    obj_param = np.zeros(4)
    prev_obj_in_poc = 0
    cam = cv.VideoCapture("sorting_1.mp4")
    file = open("coordinate_data.txt", "a")

    cnt_poc_g = 0
    cnt_poc_y = 0
    cnt_poc_r = 0
    cnt_poc_b = 0
    gb_gp = 0
    yb_yp = 0
    rb_rp = 0
    bb_bp = 0
    Hue = 0
    kf = KalmanFilter()
    pred = (0, 0)
    frame_missing = 0
    prev_cord = [0, 0, 0, 0]
    backSub = cv.createBackgroundSubtractorKNN(10)

    while True:
        ret, frame_1 = cam.read()
        if pred[0] and pred[1] == 0:
            pred = (int(frame_1.shape[1] / 2), int(frame_1.shape[0] / 2))
        if ret is True:
            mask1 = color_mask(frame_1, 20, 50)
            mask2 = color_mask(frame_1, 50, 100)
            mask3 = color_mask(frame_1, 100, 150)
            mask4 = color_mask(frame_1, 150, 255)

            frame = cv.cvtColor(frame_1, cv.COLOR_BGR2GRAY)
            frame = cv.GaussianBlur(frame, (5, 5), 0)

            diff_frame = backSub.apply((mask1 | mask2 | mask3 | mask4))

            diff_frame = cv.bitwise_and(diff_frame, diff_frame, mask=(mask1 | mask2 | mask3 | mask4))
            diff_frame = cv.erode(diff_frame, (101, 101), iterations=4)
            diff_frame = cv.dilate(diff_frame, (51, 51), iterations=4)


            (cntr, _) = cv.findContours(diff_frame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnt = sorted(cntr, key=lambda x: cv.contourArea(x), reverse=True)

            for contour in cntr:
                if 650 < cv.contourArea(contour):

                    (x, y, w, h) = cv.boundingRect(contour)
                    if h - 30 < w < h + 30 and w - 30 < h < w + 30:
                        obj_param = [x, y, w, h]
                        prev_cord = obj_param
                        Hue = detect_color(frame_1, obj_param)

                    continue

            if obj_param is not None:
                frame_missing = 0
                pred = kf.predict(obj_param[0] + int(obj_param[2] / 2), obj_param[1] + int(obj_param[3] / 2))
                cv.rectangle(frame_1, (pred[0] - 25, pred[1] - 25),
                             (pred[0] + 25, pred[1] + 25), (0, 0, 255), 3)
                cv.rectangle(frame_1, (obj_param[0], obj_param[1]),
                             (obj_param[0] + 50, obj_param[1] + 50), (0, 255, 0), 3)
                obj_in_poc = obj_poc_convergence(pok,
                                                 (obj_param[0] + (obj_param[2] / 2), obj_param[1] + (obj_param[3] / 2)))

                if 60 < Hue < 90:
                    cv.line(frame_1, (obj_param[0] + int(obj_param[2] / 2), obj_param[1] + int(obj_param[3] / 2)),
                            pok[1].center,
                            (255, 0, 0), 1)
                elif 25 < Hue < 45:
                    cv.line(frame_1, (obj_param[0] + int(obj_param[2] / 2), obj_param[1] + int(obj_param[3] / 2)),
                            pok[2].center,
                            (255, 0, 0), 1)
                elif 120 < Hue < 160:
                    cv.line(frame_1, (obj_param[0] + int(obj_param[2] / 2), obj_param[1] + int(obj_param[3] / 2)),
                            pok[3].center,
                            (255, 0, 0), 1)
                elif 100 < Hue < 110:
                    cv.line(frame_1, (obj_param[0] + int(obj_param[2] / 2), obj_param[1] + int(obj_param[3] / 2)),
                            pok[4].center,
                            (255, 0, 0), 1)

                if obj_in_poc != prev_obj_in_poc:
                    if obj_in_poc == 1:
                        cnt_poc_g += 1
                        if 60 < Hue < 90:
                            gb_gp += 1
                    elif obj_in_poc == 2:
                        cnt_poc_y += 1
                        if 25 < Hue < 45:
                            yb_yp += 1
                    elif obj_in_poc == 3:
                        cnt_poc_r += 1
                        if 120 < Hue < 160:
                            rb_rp += 1
                    elif obj_in_poc == 4:
                        cnt_poc_b += 1
                        if 100 < Hue < 110:
                            bb_bp += 1
                    else:
                        dummy = 0

                    prev_obj_in_poc = obj_in_poc
                file.write(
                    str(obj_param[0]) + ";" + str(obj_param[1]) + ";" + str(pred[0]) + ";" + str(pred[1]) + "\r\n")
                obj_param = None
                print(" Green = " + str(cnt_poc_g) + " GG = " + str(gb_gp) + " Yellow = " + str(
                    cnt_poc_y) + " YY = " + str(
                    yb_yp) + " Red = " + str(
                    cnt_poc_r) + " RR = " + str(rb_rp) + " Blue = " + str(cnt_poc_b) + " BB = " + str(bb_bp))
            else:
                frame_missing += 1

                if frame_missing < 25:
                    pred = kf.predict(prev_cord[0] + int(prev_cord[2] / 2), prev_cord[1] + int(prev_cord[3] / 2))
                    cv.rectangle(frame_1, (pred[0] - 25, pred[1] - 25),
                                 (pred[0] + 25, pred[1] + 25), (0, 0, 255), 3)
                else:
                    kf = KalmanFilter()
                    pred = (int(frame_1.shape[1] / 2), int(frame_1.shape[0] / 2))
                    cv.rectangle(frame_1, (pred[0] - 25, pred[1] - 25),
                                 (pred[0] + 25, pred[1] + 25), (0, 0, 255), 3)

            cv.imshow("Localized Object", frame_1)
            if cv.waitKey(1) == ord('q'):
                break
        else:
            break

    return cnt_poc_g, cnt_poc_y, cnt_poc_r, cnt_poc_b, gb_gp, yb_yp, rb_rp, bb_bp


if __name__ == "__main__":
    pocs = create_pocs()
    cnt_poc_g, cnt_poc_y, cnt_poc_r, cnt_poc_b, gb_gp, yb_yp, rb_rp, bb_bp = main(pocs)
    tot_b_p = cnt_poc_g + cnt_poc_y + cnt_poc_r + cnt_poc_b
    tot_correct_b_p = gb_gp + yb_yp + rb_rp + bb_bp
    tot_score = ((tot_b_p + tot_correct_b_p) * 100) / 40
    print(tot_score)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # object_center = (pocs[2].center[0] + 10, pocs[2].center[1] + 600)
    # # print(pocs[2].center,pocs[4].center)
    # obj_in_poc = obj_poc_convergence(pocs, object_center)
    # print(obj_in_poc)
