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
- [ ] Q2: Training a Support Vector Machine (25 points)
- [ ] Q3: Implement a Softmax classifier (20 points)
- [ ] Q4: Two-Layer Neural Network (25 points)
- [ ] Q5: Higher Level Representations: Image Features (10 points)
- [ ] Q6: Cool Bonus: Do something extra! (+10 points)

## [Assignment 2](http://cs231n.github.io/assignments2017/assignment2/)
- [ ] Q1: Fully-connected Neural Network (25 points)
- [ ] Q2: Batch Normalization (25 points)
- [ ] Q3: Dropout (10 points)
- [ ] Q4: Convolutional Networks (30 points)
- [ ] Q5: PyTorch / TensorFlow on CIFAR-10 (10 points)
- [ ] Q6: Do something extra! (up to +10 points)

## [Assignment 3](http://cs231n.github.io/assignments2017/assignment3/)
- [ ] Q1: Image Captioning with Vanilla RNNs (25 points)
- [ ] Q2: Image Captioning with LSTMs (30 points)
- [ ] Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images (15 points)
- [ ] Q4: Style Transfer (15 points)
- [ ] Q5: Generative Adversarial Networks (15 points)

## Project
TBA Later

## Diary Section

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
