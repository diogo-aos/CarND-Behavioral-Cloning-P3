import numpy as np
import sklearn
import pickle
import sys
import glob
import cv2
import csv
import os.path


def get_data(*args, **kwargs):
    samples = get_samples()
    # samples = select_data(samples)
    print('')
    print('total samples: ', len(samples))
    input('press enter to continue...')

    from sklearn.model_selection import train_test_split
    train_samples, val_samples = train_test_split(samples, test_size=0.2)

    if kwargs.get('generator', False):
        train = generator(train_samples,
                          kwargs.get('batch_size', 128),
                          *args, **kwargs)
        val = generator(val_samples,
                        kwargs.get('batch_size', 128),
                        *args, **kwargs)
        ret_train = train
        ret_val = val


    else:
        X_train, y_train = get_raw_data(train_samples,
                                        *args, **kwargs)
        X_val, y_val = get_raw_data(val_samples,
                                    *args, **kwargs)
        ret_train, ret_val = (X_train, y_train), (X_val, y_val)
    return ret_train, ret_val, len(samples)

def generator(samples, batch_size=128, *args, **kwargs):
    #print(samples)
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train, y_train = get_raw_data(batch_samples,
                                            *args, **kwargs)
            yield sklearn.utils.shuffle(X_train, y_train)


def get_samples(*args, **kwargs):
    load_to_mem = False
    paths_start = None
        
    if '-mem' in sys.argv:
        print('loading all data to memory...')
        load_to_mem = True
        paths_start = sys.argv.index('-mem') + 1


    elif '-gen' in sys.argv:
        print('will create generator...')
        load_to_mem = True
        paths_start = sys.argv.index('-gen') + 1

    else:
        print('-mem or -gen must be supplied with paths')
        sys.exit(0)

    arg_paths = sys.argv[paths_start:]
    data_paths = get_paths(arg_paths)
    samples = get_sample_list(data_paths)

    print('')
    print('paths that made it:')
    for p in data_paths:
        print(p)

    return samples


def get_paths(arg_paths):
    data_paths = []
    for p in arg_paths:
        globbed = glob.glob(p)
        data_paths.extend(globbed)

    for p in data_paths.copy():
        if not os.path.exists(p):
            print('path does not exist: {}'.format(p))
            data_paths.remove(p)

        if not os.path.exists(os.path.join(p, 'IMG')) or \
           not os.path.exists(os.path.join(p, 'driving_log.csv')):
            print('not valid data path: {}'.format(p))
            data_paths.remove(p)

    data_paths = [os.path.abspath(p) for p in data_paths]


    return data_paths


def get_sample_list(data_paths):
    samples = []
    for p in data_paths:
        with open(os.path.join(p, 'driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                  samples.append((p, line))  # p is necesary to reconstruct full path
    return samples


def select_data(samples, bins=20, interact=False):
    def iprint(s):
        if interact:
            print(s)
    delimiters = np.linspace(-1, 1, bins + 1)
    sample_div = []
    get_angle = lambda s: float(s[1][3])
    iprint('histogram:')
    i = 0
    for start, end in zip(delimiters[:-1], delimiters[1:]):
        div_lst = [s for s in samples if get_angle(s) >= start and get_angle(s) <= end]
        np.random.shuffle(div_lst)
        sample_div.append(div_lst)
        iprint('{} [{}, {}] \t:{}'.format(i, int(start*25), int(end*25), '*' * int(len(div_lst) * 100 / len(samples))))
        i += 1

    ret_lst = []
    for lst in sample_div:
        ret_lst.extend(lst[:1000])

    return ret_lst


def get_raw_data(samples, *args, **kwargs):
    deviation = kwargs.get('deviation', 0.0)
    if deviation is not None and not isinstance(deviation, float):
        raise TypeError('deviation must be float')
    corrected_deviation = deviation / 25.0

    crop_top, crop_bot = kwargs.get('crop_top', 0.0), kwargs.get('crop_bot', 0.0)
    if crop_top > 1.0 or crop_top < 0.0 or crop_bot > 1.0 or crop_bot < 0.0:
        raise ValueError('crop value must be between 0 and 1, faction of image')

    def crop(im):
        if crop_top == 0.0 and crop_bot == 0.0:
            return im
        h, w, c = im.shape
        top = int(h * crop_top)
        bot = h - int(h * crop_bot)
        return im[top:bot,...]


    ims = []
    angles = []
    for p, line in samples:
        cx, lx, rx, cy, ly, ry = get_sample_images(p, line, corrected_deviation)
        if cx is None or lx is None or rx is None:
            print('none imread')
            continue
        
        ims.append(cx)
        angles.append(cy)

        if deviation == 0.0:
            continue

        ims.append(lx)
        angles.append(ly)

        ims.append(rx)
        angles.append(ry)


    t_ims = []
    t_angles = []

    def apply_transform(frac, func):
        # indexes of images that will be transformed
        idx = random_sample_idx(frac, len(ims))
        for i in idx:
            x, y = func(ims[i], angles[i])
            t_ims.append(x)
            t_angles.append(y)

    if kwargs.get('random_brightness') is not None:
        apply_transform(kwargs.get('random_brightness'), random_brightness)

    if kwargs.get('random_flip') is not None:
        apply_transform(kwargs.get('random_flip'), flip_h)

    if kwargs.get('random_shadow') is not None:
        apply_transform(kwargs.get('random_shadow'), random_shadow)

    ims.extend(t_ims)
    angles.extend(t_angles)

    npims = np.array(ims)
    npangles = np.array(angles)

    # crop whole batch
    b, h, w, c = npims.shape
    top = int(h * crop_top)
    bot = h - int(h * crop_bot)
    npims = npims[:,top:bot,...]

    return npims, npangles

def get_sample_images(p, line, deviation):
    center_path = os.path.join(p, 'IMG', os.path.basename(line[0]))
    left_path = os.path.join(p, 'IMG', os.path.basename(line[1]))
    right_path = os.path.join(p, 'IMG', os.path.basename(line[2]))
    y = float(line[3])

    center_im = cv2.imread(center_path)
    left_im = cv2.imread(left_path)
    right_im = cv2.imread(right_path)

    return center_im, left_im, right_im, y, y + deviation, y - deviation


def random_sample_idx(f, total):
    n = int(f * total)
    idx = np.arange(total)
    np.random.shuffle(idx)
    return idx[:n]

def random_brightness(im, y):
    im1 = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    im1 = np.array(im1, dtype = np.float64)
    random_bright = .5 + np.random.uniform()
    im1[...,2] = im1[...,2] * random_bright
    im1[...,2][im1[...,2]>255] = 255
    im1 = np.array(im1, dtype = np.uint8)
    im1 = cv2.cvtColor(im1,cv2.COLOR_HSV2RGB)
    return im1, y

def random_shadow(image, y):
    h, w, c = image.shape
    top_y = w * np.random.uniform()
    top_x = 0
    bot_x = h
    bot_y = w * np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2) == 1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1] * random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0] * random_bright    
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image, y

def flip_h(im, y):
    im = cv2.flip(im, 1)
    return im, -y
