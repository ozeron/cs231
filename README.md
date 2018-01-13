# My process of going through cs231

[Course website](http://cs231n.github.io) – [Site repo](https://github.com/cs231n/cs231n.github.io)

I want to go through this course and will publish my progress here. Hope this process will help me to build consistency and momentum going through whole course.

## Table of Content
- [Assignment 1](#assignment-1)
- [Assignment 2](#assignment-2)
- [Assignment 3](#assignment-3)
- [Project](#project)
- [Diary](#diary-section)

-----
## [Assignment 1](http://cs231n.github.io/assignments2017/assignment1/)
- [X] Q1: k-Nearest Neighbor classifier (20 points) [17 Dec 2017](#17122017--set-up--k-nearest-neighbor-classifier)
- [X] Q2: Training a Support Vector Machine (25 points) [19 Dec 2017](#19122017--implemented-svm-naive-gradient-and-vectorized-loss) | [20 Dec 2017](#20122017--implemented-vectorize-svm-and-cross-validation)
- [X] Q3: Implement a Softmax classifier (20 points) [23 Dec 2017]() | [24 Dec 2017]()
- [X] Q4: Two-Layer Neural Network (25 points) [26 Dec 2017](#26122017--implemented-vectorize-svm-and-cross-validation)
- [X] Q5: Higher Level Representations: Image Features (10 points) [26 Dec 2017](#26122017--implemented-vectorize-svm-and-cross-validation)
- [ ] Q6: Cool Bonus: Do something extra! (+10 points)
- [ ] Recap: SVM Vectorizing Gradients

## [Assignment 2](http://cs231n.github.io/assignments2017/assignment2/)
- [X] Q1: Fully-connected Neural Network (25 points) [27 Dec 2017](#27122017--implemented-fully-connected-neural-network)
- [X] Q2: Batch Normalization (25 points) [29 Dec 2017](#29122017--implemented-batch-normalization-neural-network)
- [ ] Q2A: Implement Batch Normalization alternative gradient
- [X] Q3: Dropout (10 points) [27 Dec 2017](#27122017--implemented-fully-connected-neural-network)
- [X] Q4: Convolutional Networks (30 points)[31 Dec 2017](#31122017--implemented-convolutional-network)
- [X] Q5: PyTorch / TensorFlow on CIFAR-10 (10 points)[02 Jan 2018](#02012018--pytorch)
- [ ] Q6: Do something extra! (up to +10 points)

## [Assignment 3](http://cs231n.github.io/assignments2017/assignment3/)
- [X] Q1: Image Captioning with Vanilla RNNs (25 points)[06 Jan 2018](#06012018--rnn-and-lstm)
- [X] Q2: Image Captioning with LSTMs (30 points)[06 Jan 2018](#06012018--rnn-and-lstm)
- [ ] Q2A: Implement good captioning model
- [X] Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images (15 points) [08 Jan 2018](#08012018--network-visualization)
- [X] Q4: Style Transfer (15 points) [13 Jan 2018](#13012018--style-transfer)
- [ ] Q5: Generative Adversarial Networks (15 points)

## Project
TBA Later

## Diary Section
### 13.01.2018 – Style Transfer
Done with style transfer, interesting assignment it was a lot of fun to do this. It was quite complicated, but I manage to do it.

### 08.01.2018 – Network Visualization
Very hard at first task, I had no idea how to use PyTorch for this task, and it was almost no explicit instructions, so I just struggled a bit. Than I've groked this. Bad that I used other people solution to get idea what to do. (Especially how to pass gradients, and what to do next)

### 06.01.2018 – RNN and LSTM
Spend to days watching video lectures on CS231n, finally got to RNNs today and can complete assignments.
Good that I sit and implement them easily. I feels bad, that I can do assignment so easily, it must be harder,
thanks for a lot of help functions from creators of course.

Implementing LSTMs was pretty hard, I spent quite a lot of time even have drawn computation graph. Also when I was ready to give up and look for solutions in Internet. I'm like "Ok, one more time and I'm done"! Then "click" and everything works! That's just fucking magic! So I really implemented Everything I need to build LSTMs.

### 02.01.2018 – PyTorch
Ho-ho easily implemented architecture, it nicely overfit, but works well on data. And gets 75% accuracy.

### 31.12.2017 – Implemented Convolutional network
Woof! I did it! I've completed this assignment with convolutional networks it was pretty interesting.
Working with 3d slices of data and others. What should I do more It's fast implementation of this networks.
Everything works pretty cool!

### 29.12.2017 – Implemented Batch normalization Neural Network
Finally did this task, it was hard, I spent a lot of time. Main reason was lack of focus and attention.
Checking of shapes help a lot, but I struggled with final gradient of gate "–"
And I used external resources.

### 27.12.2017 – Implemented Fully-connected Neural Network
Implemented Fully-connected Neural Network with arbitary architecture. Now things get quite easy, I understand how to go through this course, so I just do assignments. Implemented Dropout.
The bad: I skipped descriptions of different optimizations techniques.

### 26.12.2017 – Implemented vectorize SVM and cross validation
Good that I figured out how to implement neural net, and in general completed first assignment.
Bad, I worked with low focus in the morning

### 20.12.2017 – Implemented vectorize SVM and cross validation
**GOOD**:
- implemented cross validation and SGD

**BAD**:
- spent to much time on vectorizing Gradients [This guide helped me alot](https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html)

### 19.12.2017 – Implemented SVM naive gradient and Vectorized Loss
**GOOD**:
- I succesfully figured out how to correctly calculate svm gradients. [Optimization course page](http://cs231n.github.io/optimization-1/) helped me a lot! I almost figured out it by my self, but missed indicator function they use
- First time in my life I understood how to write vectorized versions, It's very simple. You just copy-paste naive implementation and vectorize it step-by-step. One cycle per time. Loss vectorizing took only 25 minutes

**BAD**:
- I haven't out indicator function in gradients
- I do not had enough time to implement vectorized gradients

### 17.12.2017 – Set Up && k-Nearest Neighbor Classifier
So I added this repo and implemented k-Nearest Classifier. During this task I noticed few interesting things.
1. Half vectorized implementation work worse than 2 lops, probably it's because I do non efficient norm calculation
2. Vectorizing of L2 normalization do not seemed straightforward at first, but simple rule `(a-b)^2 = a^2 - 2*a*b + b^2` helps a lot. I had really to play to calculate correct square of matrix. But it was fun.
3. I got better at mutation matrixes with `reshape`, `hstack`, `vstack`. All this stuff just looked obvious for me today.

**GOOD**:
- easier than before to go through task, I'm like nailed it.
- I did everything by myself

**BAD**:
- I had to Google for simple hint about L2 vectorizing. It's quite obvious and I should use basic math rules.

So tommorow I'm going to close SVM task
