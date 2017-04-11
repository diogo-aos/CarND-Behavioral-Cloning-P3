import get_data
import cv2
import numpy as np

arg = ['driving_data/my_data/*']
paths = get_data.get_paths(arg)
samples = get_data.get_sample_list(paths)

from sklearn.model_selection import train_test_split
train_samples, val_samples = train_test_split(samples, test_size=0.2) 

train = get_data.generator(train_samples,
                           128,
                           random_brightness=0.5, random_flip=0.5, random_shadow=0.5,
                           deviation=5.0,
                           crop_top=0.35, crop_bot=0.15)

x,y = train.__next__()

def draw_angle(im, angle):
    im = im.copy()
    h, w, c = im.shape
    p0_y = h  # p0 is the starting point for the line
    p0_x = int(w/2)
    top = int(h/2)
    rad = np.deg2rad(angle * 25)
    p1_y = p0_y - int(top * np.cos(rad))
    p1_x = p0_x + int(top * np.sin(rad))
    line_im = cv2.line(im, (p0_x, p0_y), (p1_x, p1_y), color=100, thickness=2)

    return line_im

def vizualize_batch(x, y):
    i = 0
    while i < len(x):
        im, angle = x[i], float(y[i])
        print(i)
        imd = draw_angle(im, angle)
        imd = cv2.putText(imd, '{}/{} {:.1f}'.format(i, x.shape[0], angle*25), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, 100, 2)
        cv2.imshow('frame', imd)
        c = cv2.waitKey(0)
        if c == 113:
            break
        if c == 115:  # save image
            fn = input('filename for image (jpg): ')
            cv2.imwrite(fn + '.jpg', im)
            continue
        if c == 112:
            i -= 1
            continue
        i += 1
    cv2.destroyAllWindows()

