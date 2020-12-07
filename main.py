# Chase Arenas (1394033)
# Make up assignment doing Morphological Operators:
# Structuring Windows, Dilation/Erosion, and Open/Close

import cv2
import sys
import numpy as np

def structuringWindows(kernel):
    if kernel == "square_9":
        window = [[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]]
    if kernel == "square_25":
        window = [[1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1]]
    if kernel == "cross_5":
        window = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
    if kernel == "cross_9":
        window = [[0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [1, 1, 1, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]]
    if kernel == "circle":
        window = [[0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0]]

    return window

def dilation(img, window):
    # 0 padding, if window is 3x3 pad 1 row/column on each side, if 5x5 pad 2 rows/columns on each side
    if len(window) == 3:
        i = 1
    else:
        i = 2

    dilation_img = np.zeros([img.shape[0]+i, img.shape[1]+i], dtype=np.uint8)

    if len(window) == 3:
        padded_img = np.pad(img, ((1, 1), (1, 1)))
        print(padded_img)
        # print(window)
        # Need to start at (1,1)
        for i in range(1, padded_img.shape[0] - 1):
            for j in range(1, padded_img.shape[1] - 1):
                # iterate through kernel
                # 3x3 iterate through (i-1, j-1) -> (i+1, j+1)
                # starting i, j = 5,5
                # (a,b) = (0,0) we want to access 4,4
                # (a,b) = (0,1) we want to access 4,5

                final = None
                for a in range(len(window)):
                    for b in range(len(window)):
                        if window[a][b] == 1:
                            if final is None:
                                final = padded_img[i+a-1][j+b-1]
                            else:
                                final = final or padded_img[i+a-1][j+b-1]
                dilation_img[i][j] = final

    if len(window) == 5:
        padded_img = np.pad(img, ((2, 2), (2, 2)))
        # print(padded_img)
        # print(window)
        # Need to start at (2,2)
        for i in range(2, padded_img.shape[0] - 2):
            for j in range(2, padded_img.shape[1] - 2):
                # iterate through kernel
                # 5x5 iterate through (i-2, j-2) -> (i+2, j+2)
                final = None
                for a in range(len(window)):
                    for b in range(len(window)):
                        if window[a][b] == 1:
                            if final is None:
                                final = padded_img[i+a-2][j+b-2]
                            else:
                                final = final or padded_img[i+a-2][j+b-2]
                dilation_img[i][j] = final

    return dilation_img

def erosion(img, window):
    # 0 padding, if window is 3x3 pad 1 row/column on each side, if 5x5 pad 2 rows/columns on each side
    if len(window) == 3:
        i = 1
    else:
        i = 2

    erosion_img = np.zeros([img.shape[0]+i, img.shape[1]+i], dtype=np.uint8)
    erosion_img.fill(255)

    if len(window) == 3:
        padded_img = np.pad(img, ((1, 1), (1, 1)))
        # print(padded_img)
        # print(window)
        # Need to start at (1,1)
        for i in range(1, padded_img.shape[0] - 1):
            for j in range(1, padded_img.shape[1] - 1):
                final = None
                for a in range(len(window)):
                    for b in range(len(window)):
                        if window[a][b] == 1:
                            if final is None:
                                final = padded_img[i + a - 1][j + b - 1]
                            else:
                                final = final and padded_img[i + a - 1][j + b - 1]
                erosion_img[i][j] = final

    if len(window) == 5:
        padded_img = np.pad(img, ((2, 2), (2, 2)))
        # print(padded_img)
        # print(window)
        # Need to start at (2,2)
        for i in range(2, padded_img.shape[0] - 2):
            for j in range(2, padded_img.shape[1] - 2):
                # iterate through kernel
                # 5x5 iterate through (i-2, j-2) -> (i+2, j+2)
                final = None
                for a in range(len(window)):
                    for b in range(len(window)):
                        if window[a][b] == 1:
                            if final is None:
                                final = padded_img[i + a - 2][j + b - 2]
                            else:
                                final = final and padded_img[i + a - 2][j + b - 2]
                erosion_img[i][j] = final

    return erosion_img

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--image", dest="image", metavar="IMAGE")
    parser.add_argument("-w", "--window", dest="window",
                        help="Specify Window (square_9, square_25, cross_5, cross_9, circle)", metavar="WINDOW")
    args = parser.parse_args()
    if args.image is None:
        print("Please specify image")
        sys.exit(2)
    else:
        # Using cv2 to convert blob.png into a Binary Image
        image_name = args.image
        image = cv2.imread(image_name, 2)
        ret, binary_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        bin_image = 255 - binary_img

    if args.window is None:
        print("Please specify window (square_9, square_25, cross_5, cross_9, circle)")
        sys.exit(2)
    else:
        struct_window = structuringWindows(args.window)
        # print(structuringWindows(args.window))

    output_dir = 'output/'
    # 3 iterations
    output = dilation(bin_image, struct_window)
    output2 = dilation(output, struct_window)
    output3 = dilation(output2, struct_window)

    # kernel = np.ones((5, 5), np.uint8)
    # output = cv2.erode(bin_image, kernel, iterations=3)

    output_name = output_dir+image_name+'_dilation.png'
    cv2.imwrite(output_name, output3)

    # 3 iterations
    output4 = erosion(bin_image, struct_window)
    output5 = erosion(output4, struct_window)
    output6 = erosion(output5, struct_window)

    # kernel = np.ones((5, 5), np.uint8)
    # output = cv2.dilate(bin_image, kernel, iterations=3)

    output_name = output_dir+image_name+'_erosion.png'
    cv2.imwrite(output_name, output6)

    # OPEN
    # kernel = np.ones((5, 5), np.uint8)
    # output = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)

    # output_name = output_dir+image_name+'_open.png'
    # cv2.imwrite(output_name, output)

    # CLOSE
    # kernel = np.ones((5, 5), np.uint8)
    # output = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)

    # output_name = output_dir+image_name+'_close.png'
    # cv2.imwrite(output_name, output)


if __name__ == "__main__":
    main()