# Starting point
02-04-2017 15:01

The starting model is LeNet. I've augmented the data to include the left and
right images with a deviation. I've tried a deviation of 0.02 and 0.04 but
non yield good results. I'm also using gathered data only. It includes 2
counter-clockwise laps and 1 clockwise lap around the easy circuit. It also
includes small data on recovering from the sides.

So far I have not had any success. I didn't get the results shown in class
for LeNet.

I think one of the reasons is that the fully connected layers are too small for
the ammount of data the convolutional part is giving out.

# Fixing LeNet
02-04-2017 17:19
Turns out only the filename was Lenet - the network itself was just a the 
input image flattened and fed into a single output neuron.

I implemented LeNet, but the car just goes forward, never steering.


# Redoing data loading and Nvidia net
02-04-2017 23:17

Redid the mechanism I was using to load the data. Now I load directly from the data folders created by the 
simulator. I was worried that the bad results were someone related to the intermediate step of saving the data in
a different format. It doesn't seem to be the case as the bad result of not steering continues.

I implemented the network used in the NVIDIA end-to-end steering paper. I trained it on the udacity data, but I
still got bad results. Now I'm traning on augmented data (all cameras with flipping).

# I did it my data
03-04-2017 00:15

I trained the nvidia net with my data and got a BIG improvement. Two faults detected: after the bridge it goes 
to the dirt and on the sharp curve it does not steer enough to the right and goes to the lake.

I'm training the network and cropping the top now. I hope that it will get less "distracted" and does not go into
the dirt.

# Update
03-04-2017 09:09

Added data for hard track and more recovery segments. Performance improved. Still no preprocessing other than
normalization and cropping.

# Histogram
05-04-2017 15:11

I computed the histogram of the input data and observed there was a big imbalance in samples per angle.
The vast majority of the samples have a steering angle basicly going forward.

I limited the amount of sample on the range [-0.25, 0] and [0, 0.25] (0.25 is 6.25 degrees) to 1000 and it helped on the tight curve, but it also made it crash on the bridge, at the end. I'm shufling the data before sampling these intervals, so I might be
loosing good data on the bridge. I also trained with a 5 degree deviation (in contrast to the 1deg used before).

I also noted that the amound of data steering to the right is less than its counterpart. I've collected more data steering to the right.

To try:
 - train on whole data with new deviation
 - reduce the size of the interval cut, e.g. (go from 6.25 to something like 3 degrees)
 - get 400 samples from each angle interval, making the dataset a lot smaller but much more balanced as well

# train on whole data with 5deg deviation
05-04-2017 15:47

Using the whole dataset with the new deviation made the car go around the track fine, both clockwise and 
anticlockwise. I also increased the speed to 20 to make it faster. I tried 30, but the car crashed more easily (I
suspect there are not enough refreshes to make it stable).

# got 500 samples from 20 angle intervals
05-04-2017 16:46

The car went through the track both directions. It seemed to have a greater tendency to "hug" markers in some portions
of the road. At higher speeds it was definetely more unstable.

# more work
06-04-2017 15:43

I had thought about just delivering the project as it is, but after reading on the internet about people doing 
random transformations with their data, I've decided to experiment some of those things to see if I can improve
the performance of the model.

First I need to vizualize the result of the cropping I'm doing. Then some things to try:

random brightness adjustments, artificial shadows, and horizon shifts


# data transformations
06-04-2017 16:47

I've added code to make random transformations to the data before training. Horizontal
flipping is now a random transformation, as is brightness and shadows. I choose a fraction
of the data to which each transformation is applied. That fraction is then added to the 
dataset before training.

I noticed that the training loss continues to decrease but validation loss oscilates. I've
added 2 dropout layers (nvidia2.py) and am now waiting for results. It takes longer to 
converge now.

# data transformations
10-04-2017 18:37

THe dropout experience was a failure - didn't work good. I returned to normal NVIDIA arch.

Now I'm rerunning model training for different dataset augmentations (fractions of data, deviations) and being systematic so that I can track what works and not.

As of now I've got a model that does one lap good and then crashes after the bridge (right after the offroad section) on the second lap (weird...).

I've trained a model on all the data with transformations and a 33% fraction to each transformation. Deviation of 5º. Will run that model when I get good internet, probably not today.

I've left a model training with the same specs but only on data of the first track.
