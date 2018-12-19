# Strange Runes

This is a Python 3.6 package for solving the [image registration problem](https://en.wikipedia.org/wiki/Image_registration)
using TensorFlow. Image registration involves trying to align two images with one another so that they share one coordinate system.

---

## 1. Usage Example

Strange runes uses optimization algorithms to minimize the distance between some misaligned "floating" image and some
aligned "fixed" image. Because this is a highly nonlinear problem two additional steps are taken. Firstly, the distance
between the two images is not computed on the raw images, but rather on features extracted from the raw images by a
trained siamese neural network. Secondly, due to the non-linearity of the search space global optimization algorithms
including a greedy algorithm, a [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm), and a
[particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) algorithm. Details on the
siamese neural network and global search algorithms can be found below. For now, here's an example of a result
produced with the package.

### 1.1 Results Example

By minimizing the sum squared error of the features extracted by the siamese neural network for the floating and fixed
image the particle swarm was able to determine that the best affine transformation was:

* Scale X percent = 92.7%
* Scale Y percent = 100%
* Rotate radians = 3.005
* Shear radians = 0.043
* Translate X pixels = -26
* Translate Y pixels = -15

More details from this specific example are provided in the details section further down. In the details the specific
features are shown as well as the results learnt by the genetic algorithm and the greedy search algorithm.

![PSO Image Alignment Result](http://www.turingfinance.com/wp-content/uploads/2018/12/example_out.png "PSO Image Alignment Result")

### 1.2 Python Code

```python
warnings.filterwarnings("ignore")

# Specify the path to the images we want to work with
pth = os.path.join("/", "home", "stuart", "GitHub", "strange_runes",
                   "strange_runes", "data", "source_images")

# Create a similarity neural network and load weights for it
similarity_net = AugSimNet(name="cnn_single", path_to_images=pth)
similarity_net.load_trained_weights()

# Randomly select a test image to demonstrate search on
image = similarity_net.load_image(similarity_net.images["test"][0])
image_goal, image_curr, true_move_vector = \
    similarity_net.warp_image(image=image, method="normal")

# Plot the gloating and goal images - the problem
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))
f.suptitle("IMAGE REGISTRATION", fontsize=16)
ax0.imshow(image_goal)
ax0.set_title("GOAL IMAGE")
ax1.imshow(image_curr)
ax0.set_title("FLOATING IMAGE")
plt.savefig("problem.png")
plt.close()

# Plot the feature vector. This shows us what
# goes into the NN's softmax layer
similarity_net.plt_features(
    image_a=image_goal, image_b=image_curr,
    file_name="feature_vector")

# Plot the feature maps extracted by the
# similarity network for the two images. This
# is what we are tyring to align with search
similarity_net.plt_feature_maps(
    input_image=image_curr,
    file_name="feature_maps_curr")
similarity_net.plt_feature_maps(
    input_image=image_goal,
    file_name="feature_maps_goal")

# Plot the "evolution" of the features extracted
# by the similarity network. This is useful for
# understanding the neural network itself.
similarity_net.plt_all_feature_maps(
    arch_name="deep_cnn", input_image=image_curr,
    file_name="feature_evolution_curr")
similarity_net.plt_all_feature_maps(
    arch_name="deep_cnn", input_image=image_goal,
    file_name="feature_evolution_goal")

# --- GREEDY SEARCH ---

print("=" * 120)
print("GREEDY SEARCH")

# Create a greedy search algorithm
greed = GreedyImageSearch(
    fixed_image_goal=image_goal,
    floating_image_start=image_curr,
    similarity_net=similarity_net,
    similarity_metric="feature_maps")

# Search for an optimal transformation greedily
greed_optimal, greed_score = greed.search()
greed.plt_result(greed_optimal, name="greedy.png")
print("GREEDY ALGORITHM SCORE:", greed_score)

# --- GENETIC SEARCH ---

print("=" * 120)
print("GENETIC ALGORITHM")

# Create a genetic search algorithm
ga = GeneticImageSearch(
    fixed_image_goal=image_goal,
    floating_image_start=image_curr,
    similarity_net=similarity_net,
    similarity_metric="feature_maps")

# Evolve an optimal image transformation
ga_optimal, ga_score = ga.search()
ga.plt_result(ga_optimal, name="genetic.png")
print("GENETIC ALGORITHM SCORE:", ga_score)

# --- PARTICLE SEARCH ---

print("=" * 120)
print("PARTICLE SWARM")

# Create a particle swarm search algorithm
pso = ParticleImageSearch(
    fixed_image_goal=image_goal,
    floating_image_start=image_curr,
    similarity_net=similarity_net,
    similarity_metric="feature_maps")

# Search for an optimal transformation by PSO
pso_optimal, pso_score = pso.search()
pso.plt_result(pso_optimal, name="particle.png")
print("PARTICLE SWARM SCORE:", pso_score)
```

---

## 2. Details of Approach

### 2.1 Automatic Feature Extraction

This section details the approach taken to extract features for alignment.

#### 2.1.1 Convolutional Neural Networks

Neural networks are end-to-end machine learners meaning that they are given data and will automatically extract salient
features from the image which are helpful for the task assigned to the neural network. In deep neural networks this
feature extraction is performed by successive linear functions passed through a nonlinear activation function.

Convolutional neural networks are a class of neural networks often used for computer vision tasks. They do not use
fully-connected, otherwise known as dense, layers. They use a kind of locally connected layer which can be thought of
as a hard prior on the weights of an otherwise fully connected layer. This hard prior is adopted for two reasons:

1. It makes the learning task tractable because far fewer parameters are required.
2. It forces the neural network to pay attention to local correlations that exist images.

The amount of local correlations captured by a layer is called "the receptive field". In order for the convolutional
neural network to learn from global correlations in the image, the receptive field must increase through the layers.
This can be achieved by changing the parameters of the layers through the convolutional neural network or downscaling
the image as it passes through the layers of the neural network. This idea is illustrated below in the classic VGG
neural network architecture:

![VGG Neural Network](https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png "VGG Architecture")

#### 2.1.2 Siamese Neural Networks

A siamese neural network is a special kind of convolutional neural network in which:

1. Two images, image A and image B, are inputted instead of one,
2. The weights used to extract features from image A and image B are the same,
3. The absolute distance between image A's features and image B's features is computed, and
4. This is passed into a dense layer which outputs the probability that the images are the same or different

The reason why a siamese neural network is used is because the network _should_ (in theory) learn the salient features
that are useful for distinguishing between images of the same thing and images of another thing. My hypothesis was that
these features might provide useful information for the image registration problem. This appears to be true.

![Siamese Network](http://www.turingfinance.com/wp-content/uploads/2018/12/siamese_nnet.png "Siamese Network")

_NOTE: image sourced from: https://github.com/akshaysharma096/Siamese-Networks_

#### 2.1.3 Ensembling

In strange runes the architecture of the neural network is specified as a JSON file:

```json
{
    "features": {
        "deep_cnn": {
            "repeats": 3,
            "layers": [
                {
                    "class": "conv_2d",
                    "dilation_rate": 1,
                    "filters": "auto",
                    "strides": 1,
                    "use_bias": true,
                    "activation": "relu",
                    "kernel_initializer": "glorot_uniform",
                    "bias_initializer": "zeros",
                    "kernel_size": [
                        3,
                        3
                    ],
                    "padding": "same",
                    "l2_regularization": 0.005,
                    "bias_constraint": "nonneg"
                },
                {
                    "class": "conv_2d",
                    "dilation_rate": 2,
                    "filters": "auto",
                    "strides": 1,
                    "use_bias": true,
                    "activation": "relu",
                    "kernel_initializer": "glorot_uniform",
                    "bias_initializer": "zeros",
                    "kernel_size": [
                        3,
                        3
                    ],
                    "padding": "same",
                    "l2_regularization": 0.005,
                    "bias_constraint": "nonneg"
                },
                {
                    "class": "max_pooling_2d",
                    "pool_size": [
                        2,
                        2
                    ]
                }
            ]
        }
    }
}
```

It is possible to define multiple different convolutional neural network architectures in this JSON file. If done,
the siamese network network will extract features from each sub neural network with fixed weights for image A
and image B, concatenate the differences between the features, and feed this into the final dense layer. This allows
for easy ensembling and hypothesis testing regarding which types of convolutional neural networks are most appropriate.

For example you can create a siamese neural network which looks at the combined the features from a 3x3 deep
convolutional neural network and a 5x5 deep convolutional neural network as follows:

```json
{
    "features": {
        "deep_3x3_cnn": {
            "repeats": 3,
            "layers": [
                {
                    "class": "conv_2d",
                    "dilation_rate": 1,
                    "filters": "auto",
                    "strides": 1,
                    "use_bias": true,
                    "activation": "relu",
                    "kernel_initializer": "glorot_uniform",
                    "bias_initializer": "zeros",
                    "kernel_size": [
                        3,
                        3
                    ],
                    "padding": "same",
                    "l2_regularization": 0.005,
                    "bias_constraint": "nonneg"
                },
                {
                    "class": "conv_2d",
                    "dilation_rate": 2,
                    "filters": "auto",
                    "strides": 1,
                    "use_bias": true,
                    "activation": "relu",
                    "kernel_initializer": "glorot_uniform",
                    "bias_initializer": "zeros",
                    "kernel_size": [
                        3,
                        3
                    ],
                    "padding": "same",
                    "l2_regularization": 0.005,
                    "bias_constraint": "nonneg"
                },
                {
                    "class": "max_pooling_2d",
                    "pool_size": [
                        2,
                        2
                    ]
                }
            ]
        },
        "deep_5x5_cnn": {
            "repeats": 3,
            "layers": [
                {
                    "class": "conv_2d",
                    "dilation_rate": 1,
                    "filters": "auto",
                    "strides": 1,
                    "use_bias": true,
                    "activation": "relu",
                    "kernel_initializer": "glorot_uniform",
                    "bias_initializer": "zeros",
                    "kernel_size": [
                        5,
                        5
                    ],
                    "padding": "same",
                    "l2_regularization": 0.005,
                    "bias_constraint": "nonneg"
                },
                {
                    "class": "conv_2d",
                    "dilation_rate": 2,
                    "filters": "auto",
                    "strides": 1,
                    "use_bias": true,
                    "activation": "relu",
                    "kernel_initializer": "glorot_uniform",
                    "bias_initializer": "zeros",
                    "kernel_size": [
                        5,
                        5
                    ],
                    "padding": "same",
                    "l2_regularization": 0.005,
                    "bias_constraint": "nonneg"
                },
                {
                    "class": "max_pooling_2d",
                    "pool_size": [
                        2,
                        2
                    ]
                }
            ]
        }
    }
}
```

#### 2.1.4 Training Process

A slightly different training process was adopted by Strange Runes. This worked as follows:

* For 1 .. training_sessions
    * For 1 .. images_per_training_session
        * Sample an image at random, A, from the correct set (train / valid)
        * Sample two random affine transformation, f and g
        * Produce A_1 = f(A) and A_2 = g(A)
        * Sample a different image at random, B, from the correct set
        * Sample two random affine transformation, h and l
        * Produce B_1 = h(B) and B_2 = l(B)
        * Add pattern [A_1, A_2] with response "same" to batch
        * Add pattern [B_1, B_2] with response "same" to batch
        * Add pattern [A_1, B_1] with response "different" to batch
        * Add pattern [A_2, B_2] with response "different" to batch
    * For 1 .. num_epochs_per_session
        * Train the neural neural network

This training process has two benefits. Firstly, it uses data augmentation to amplify the amount of training data
available. This has the effect of reducing the probability of overfitting (memorizing) to the images. Secondly, the
neural network needs to learn features in the data that are invariant to affine transformations ... on that note,
it is not clear to me whether this is actually good or bad for the task at hand, classical more location sensitive
features may perform better?

#### 2.1.5 Visualization of Features

Since the features are quite important to this task, I added the ability to visualize the feature maps learnt by
the individual convolutional neural networks in the ensemble as well as the ability to visualize how the feature maps
evolve as you go deeper into the neural network. The images that follow are the features extracted on the floating
image shown earlier.

![Layer 0 Features](http://www.turingfinance.com/wp-content/uploads/2018/12/feature_evolution_curr_depth0.png "Layer 0 Features")
![Layer 1 Features](http://www.turingfinance.com/wp-content/uploads/2018/12/feature_evolution_curr_depth1.png "Layer 1 Features")
![Layer 2 Features](http://www.turingfinance.com/wp-content/uploads/2018/12/feature_evolution_curr_depth2.png "Layer 2 Features")
![Layer 3 Features](http://www.turingfinance.com/wp-content/uploads/2018/12/feature_evolution_curr_depth3.png "Layer 3 Features")
![Layer 4 Features](http://www.turingfinance.com/wp-content/uploads/2018/12/feature_evolution_curr_depth4.png "Layer 4 Features")
![Layer 5 Features](http://www.turingfinance.com/wp-content/uploads/2018/12/feature_evolution_curr_depth5.png "Layer 5 Features")
![Layer 6 Features](http://www.turingfinance.com/wp-content/uploads/2018/12/feature_evolution_curr_depth6.png "Layer 6 Features")
![Layer 7 Features](http://www.turingfinance.com/wp-content/uploads/2018/12/feature_evolution_curr_depth7.png "Layer 7 Features")

### 2.2 Search Algorithm Details

Three search algorithms were implemented for this task:

1. A Greedy Search Algorithm
2. A [Genetic Search Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) (GA)
3. A [Particle Swarm Search Algorithm](https://en.wikipedia.org/wiki/Particle_swarm_optimization) (PSO)

All three are "global" search algorithms in the sense that they do not rely on any local gradient information in
order to arrive at a locally optimal solution (finding the true global optimum is NP-hard to cannot be guaranteed).

These algorithms were chosen because a (very brief) review of some literature on image registration indicated that
this is a non-convex and multi-modal search problem. What follows is a brief explanation of how these algorithms work.

#### 2.2.1 Greedy Search Algorithm

The greedy search algorithm essentially loops through each of the parameters in the affine transformation and moves
them up or down until the score stops improving. The amount that the parameter value is moved up or down is called
the step size. The algorithms successively tries smaller and smaller step sizes. This has the effect of shifting the
focus of the algorithm from exploration at the start of the routine to exploitation at the end of the routine. This is
a desirable property.

* Set the starting transformation, f
* For step_size in [50%, 25%, ..., 1%, 0.5%]
    * For parameter in f
        * improving = True
        * While improving
            * Generate a random left step, f-
            * Generate a random right step, f+
            * If score(f-) < score(f+)
                * If score(f-) >= score(f)
                    * improving = False
                * Else
                    * Update f to f-
            * Else
                * If score(f+) >= score(f)
                    * improving = False
                * Else
                    * Update f to f+
* return f

##### Solution Found

![Greedy Search Solution](http://www.turingfinance.com/wp-content/uploads/2018/12/greedy_solution.png "Greedy Search Solution")

#### 2.2.2 Genetic Search Algorithm

Genetic Algorithms are a nature-inspired, population-based algorithm based off of the principles of evolution namely,
(1) the fittest individuals survive through time, (2) individuals recombine to produce offspring, and (3) mutations
occur at random throughout the population and time. These principles can be applied to any optimization problem in
order to evolve fitter and fitter solutions to the specified optimization problem. For image registration this is
implemented as follows,

* Sample a population of random affine transformations, F
* For epoch 1 .. N
    * Rank the individuals in the population
    * For individual, f, in population, F
        * If f has not survived
            * Pick parent 1, f', from F that has survived
            * Pick parent 2, f'', from F that has survived
            * Replace f in population with a combination of f' and f''
        * If f is not the fittest individual:
            * If uniform(0, 1) < mutation_rate:
                * Generate a random affine transformation, g
                * Replace f in population with a combination of f and g
* If greedy:
    * Apply greedy search on fittest individual, f*
* return f*

This optimization process is illustrated in the diagram below.

![Genetic Algorithm](http://www.turingfinance.com/wp-content/uploads/2014/04/Genetic-Algorithm1.png "Genetic Algorithm")

##### Solution Found

![Genetic Search Solution](http://www.turingfinance.com/wp-content/uploads/2018/12/genetic_solution.png "Genetic Search Solution")

#### 2.2.3 Particle Swarm Search Algorithm

The Particle Swarm Optimization (PSO) is another nature-inspired, _swarm_ based algorithm. This algorithm constructs a
swarm of candidate solutions to the optimization problem. Each candidate solution contains: (1) its current location,
(2) its previously most-optimal location, and (3) its current velocity. The algorithm works by iteratively updating
the velocity of each particle such that it is simultaneously directed towards its own historical personal best location
and the global best position found by the swarm. This dual update has the effect of increasing the algorithms ability
to explore the search space without compromising the algorithms ability to exploit a locally-optimal solution. For
image registration this is implemented as follows,

* Sample a swarm of affine transformations, F
* Set swarms personal bests to F
* Initialize velocities close to zero for swarm
* For epoch 1 .. N
    * For particle, f, in swarm, F
        * If score(f) < score(f_best)
            * Update personal best location for f
        * If score(f) < best_score:
            * Update global best location, f*, for F
    * For particle, f, in swarm, F
        * if f is not f*
            * Compute the social component (move towards f*)
            * Compute the cognitive component (move towards f_best)
            * Update the velocity of f by the social and cognitive components
            * Update the location of f by the velocity of f
* If greedy:
    * Apply greedy search on best individual, f*
* return f*

This optimization process is illustrated in the diagrams below.

![Particle Search Solution](http://www.turingfinance.com/wp-content/uploads/2013/12/Particle-Swarm-Optimization.png "Particle Swarm Example")
![Particle Search Solution](http://www.turingfinance.com/wp-content/uploads/2013/12/Particle-Swarm-Optimization-Portfolio-Optimization.png "Particle Swarm Step 1")
![Particle Search Solution](http://www.turingfinance.com/wp-content/uploads/2013/12/Particle-Swarm-Optimization-Portfolio-Optimization-2.png "Particle Swarm Step 2")
![Particle Search Solution](http://www.turingfinance.com/wp-content/uploads/2013/12/Particle-Swarm-Optimization-Portfolio-Optimization-3.png "Particle Swarm Step 3")

##### Solution Found

![Particle Search Solution](http://www.turingfinance.com/wp-content/uploads/2018/12/particle_solution.png "Particle Search Solution")

#### 2.2.4 Comparison of Algorithms (Notes)

My first observation is that both the genetic algorithm and the particle swarm algorithm consistently beat the greedy
search algorithm. This indicates that the search space is probably non-convex and multi modal as others have stated.

My second observation is that the particle swarm generally beats the genetic algorithm. This is consistent with my
results on other problems and is probably due to the increased "reachability" of the algorithm. All of the variation
in the solutions in the genetic algorithm comes from the initialization of the population and from mutation. As such
genetic algorithms tend to get "stuck" on specific values. This works well for discrete optimization problems. For
continuous optimization problems such as the image registration problem, particle swarm's vectorized approach is
expected to be better.

My third observation is that the greedy algorithm is the least computationally expensive (< 20 seconds per alignment),
the genetic algorithm is the second least computationally expensive (~2 minutes per alignment), and the particle
swarm optimization is the most computationally expensive (~4 minutes per alignment). However, the implementation of
both the genetic algorithm and particle swarm can be tuned to reduce computational overhead and this implementation is
sub-optimal and could be sped up by a factor of at least 6 - 8x by moving to a parallel computing approach.

---

## 3. Shortcomings and Next Steps

The major shortcoming of this approach is computational complexity and runtime. That having been said, a number of
small improvements could improve the runtime significantly. These should be the next steps in this project.

1. Squeeze the neural network for faster inference and faster search
2. Implement the image transformations in Cython as they are the bottleneck
3. Implemented distributed versions of both the Genetic Algorithm and Particle Swarm Algorithm

After these improvements I suspect the solution would be workable for a production environment ... after some testing
of course. Due to time constraints no tests are included in this package, but everything has been eyeballed :-).

