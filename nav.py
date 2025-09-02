import freenect
import cv2
import numpy as np

def nothing(x):
    pass

# Create windows
cv2.namedWindow('Video')
cv2.moveWindow('Video', 5, 5)
cv2.namedWindow('Navig', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Navig', 400, 100)
cv2.moveWindow('Navig', 700, 5)

# Define kernel and load image
kernel = np.ones((5, 5), np.uint8)
imn = cv2.imread('blank.bmp')

# Create trackbars
cv2.createTrackbar('val1', 'Video', 37, 1000, nothing)
cv2.createTrackbar('val2', 'Video', 43, 1000, nothing)
cv2.createTrackbar('bin', 'Video', 20, 50, nothing)
cv2.createTrackbar('erode', 'Video', 4, 10, nothing)
cv2.createTrackbar('epsilon', 'Video', 1, 100, nothing)
cv2.createTrackbar('spacing', 'Video', 30, 100, nothing)

def pretty_depth(depth):
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

def RegionCheck(foo, ListPath):
    if (foo <= 130) and (ListPath[0] != 0):
        ListPath[0] = 0
    if (foo > 130) and (foo <= 320) and (ListPath[1] != 0):
        ListPath[1] = 0
    if (foo > 320) and (foo <= 510) and (ListPath[2] != 0):
        ListPath[2] = 0
    if (foo > 510) and (ListPath[3] != 0):
        ListPath[3] = 0
    return ListPath

def imgshow(ListPath, t, imn, Winname):
    if ListPath[1:3] == [1, 1]:
        imn = cv2.imread(f"{t}frwd.bmp")
    elif ListPath[2:4] == [1, 1]:
        imn = cv2.imread(f"{t}right.bmp")
    elif ListPath[0:2] == [1, 1]:
        imn = cv2.imread(f"{t}left.bmp")
    else:
        imn = cv2.imread(f"{t}back.bmp")
    cv2.imshow(Winname, imn)

print('Press \'b\' in window to stop')

while True:
    imn = cv2.imread('blank.bmp')
    cv2.imshow('Navig', imn)
    flag120 = [1, 1, 1, 1]
    flag140 = [1, 1, 1, 1]
    f14 = 0
    f12 = 0
    f10 = 0
    f8 = 0

    # Get depth frame
    dst = pretty_depth(freenect.sync_get_depth()[0])
    cv2.flip(dst, 0, dst)
    cv2.flip(dst, 1, dst)
    
    # Draw a rectangle on the depth image
    cv2.rectangle(dst, (0, 0), (640, 480), (40, 100, 0), 2)
    
    # Get trackbar positions
    binn = cv2.getTrackbarPos('bin', 'Video')
    e = cv2.getTrackbarPos('erode', 'Video')
    dst = (dst // binn) * binn
    dst = cv2.erode(dst, kernel, iterations=e)
    
    v1 = cv2.getTrackbarPos('val1', 'Video')
    v2 = cv2.getTrackbarPos('val2', 'Video')
    edges = cv2.Canny(dst, v1, v2)
    
    # Find contours and draw them
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, (0, 0, 255), -1)
    
    spac = cv2.getTrackbarPos('spacing', 'Video')
    rows, cols = dst.shape

    # Iterate through grid points
    for i in range(0, rows, spac):
        for j in range(0, cols, spac):
            cv2.circle(dst, (j, i), 1, (0, 255, 0), 1)
            depth_value = dst[i, j]
            if depth_value == 80:
                f8 = 1
                cv2.putText(dst, "0", (j, i), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 20), 2)
                cv2.putText(dst, "Collision Alert!", (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, 2, 1)
                imn = cv2.imread("Collision Alert.bmp")
                cv2.imshow('Navig', imn)
            elif depth_value == 100:
                f10 = 1
                cv2.putText(dst, "1", (j, i), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 20), 2)
                cv2.putText(dst, "Very Close proximity. Reverse", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, 2, 1)
                if not f8:
                    imn = cv2.imread("VCP Reverse.bmp")
                    cv2.imshow('Navig', imn)
            elif depth_value == 120:
                f12 = 1
                cv2.putText(dst, "2", (j, i), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 20), 2)
                flag120 = RegionCheck(j, flag120)
                if not f8 and not f10:
                    imgshow(flag120, 120, imn, 'Navig')
            elif depth_value == 140:
                f14 = 1
                cv2.putText(dst, "3", (j, i), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 20), 1)
                flag140 = RegionCheck(j, flag140)
                if not f8 and not f10 and not f12:
                    imgshow(flag140, 140, imn, 'Navig')

    # Draw lines on the depth image
    cv2.line(dst, (130, 0), (130, 480), 0, 1)
    cv2.line(dst, (320, 0), (320, 480), 0, 1)
    cv2.line(dst, (510, 0), (510, 480), 0, 1)
    cv2.imshow('Video', dst)
    
    # Exit the loop on pressing 'b'
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cv2.destroyAllWindows()

