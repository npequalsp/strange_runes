import gc
import json
import os
import random

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from cv2 import resize
from strange_runes.image_tools import ImageMover
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalMaxPooling2D, \
    Concatenate, AveragePooling2D, Dropout, Lambda, SeparableConv2D, MaxPooling3D, BatchNormalization, DepthwiseConv2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow_probability.python.layers import Convolution2DReparameterization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AugSimNet(object):
    """
    This class contains code for creating AugSimNet (Augmented Siamese Networks) objects. This object exposes methods
    which allow the user to train, evaluate, and visualize a model to perform the following task:

    * Choose image A and B where A != B
    * Apply two random affine transforms to A to produce A' and A''
    * Apply two random affine transform to B to produce B' and B''
    * Construct the following training pairs:
      - [A'  , A''] -> 1 (same)
      - [B'  , B''] -> 1 (same)
      - [A'  , B' ] -> 0 (different)
      - [B'' , A''] -> 0 (different)
    * Learn a function that outputs the probability that two images are the same.

    This is the classical Siamese neural network problem. However, AugSimNet offers some useful extensions on a typical
    siamese neural network namely: data augmentation, built-in ensembling, ablation studies, and visualization.

    Why augment the training data?
    ------------------------------

    Neural networks are generally over-parametrized w.r.t the number of and dimensionality of the training data. Put
    differently, neural networks have significant informational capacity also known as expressiveness.

    In theory this means that it would be possible for the neural network to embed every image into the parameters of
    the neural network. This corresponds to "memorization" rather than "learning". There are a few ways to avoid this:

    1. Regularization - penalize the model for using more of it's capacity than it required.
    2. Dropout - randomly "switch off" activations so as to not over fit to individual features
    3. Noising - add a little bit of noise to every activation map (you can't memorize a moving target)
    4. Data Augmentation - train the model on randomly distorted images instead of the actual images
    5. Auxillary Losses - train the model to perform more than one task at a time

    Methods 1, 2, 3, and 4 can be implemented in AugSimNet. As mentioned above, data augmentation is implemented by
    default. This is to reduce the probability of the model memorizing the images ... intuitively, it's hard to
    memorize every image if you only ever see them once (because they are distorted every single time).

    Why ensemble?
    -------------

    Ensembling is when you make up a larger model from a lot of usually smaller and usually weaker models. There are
    a few benefits to this approach derived from (1) using small models and (2) using different models.

    Smaller models tend to be:

    1. Less likely to overfit ... but also more likely to underfit
    2. Must quicker to train, deploy, and are usually quicker at inference time

    Different models tend to:

    1. Be uncorrelated and therefore they provide complementary signals
    2. Be more robust in the presence of changing data

    Why do ablation studies?
    ------------------------

    Ablation studies involves switching off specific aspects of the neural network and measuring the degradation in
    the predictive accuracy of the model. The idea being that the "most important" aspects of the neural network should
    result in the worst degradation when removed. This is very useful because it allows the user to identify which
    aspects of the neural network are working and which are not. This informs the user and can elucidate the right
    path to follow for model improvement. It can also help to identify edge cases that lead to failure ahead of time.

    AugSimNet only implements a coarse-grained ablation study at the ensemble level for identifying which models in
    the ensemble are contributing the most to the model's performance. More complex ablations are possible.

    Why visualize?
    --------------

    The most underrated test in data science is the "eyeball" test. Simply by looking at the feature maps produced by
    different neural networks in the ensemble can explain weird behaviours and identify areas for improvement.

    """
    def __init__(self, name, path_to_images, *args, **kwargs):
        """
        Constructor for AugSimNet objects.

        :param name: the name of the resulting AugSimNet
        :param path_to_images: the path to the training images
        """

        # The name of the model is used to determine
        # which json file to load and which h5 file
        # (the weights) to load into the model.
        self.name = name

        # Get the path to the source directory (where the models are)
        self.path_to_here = os.path.dirname(os.path.abspath(__file__))

        # Load the configuration for this AugSimNet object - contains the ensembles
        self.path_to_config = os.path.join(self.path_to_here, self.name + ".json")
        with open(self.path_to_config) as config_data:
            config_data = json.load(config_data)
            self.sub_graphs = config_data["features"]

        # Attributes relating to data and random warps
        self.path_to_data = path_to_images
        self.train_valid_test_split = (0.60, 0.30)
        self.images = {"train": [], "valid": [], "test": []}
        self.pad_min_pct, self.pad_max_pct = 0.0, 0.50

        # Attributes relating to prints
        self.print_summary = False
        self.verbose = False

        # Attributes relating to the training sessions. These control
        # how many images the model sees and how many times
        self.image_size = (512, 512)
        self.images_per_training_session = 256
        self.images_per_evaluation = 128
        self.warps_per_image = 4
        self.batch_size = 16
        self.num_samples = 20
        self.training_sessions = 100
        self.epochs_per_session = 4

        self.max_scale_pct = 0.15
        self.max_rotate_degrees = 365
        self.max_translate_pixels = 128
        self.max_shear_degrees = 5

        # Store the models
        self.model = None
        self.sub_models = {}
        self.layer_names = {}

        for kw, attr in kwargs.items():
            setattr(self, kw, attr)

        # An ImageMover for manipulating images
        self.image_mover = ImageMover(
            height=self.image_size[0],
            width=self.image_size[1],
            max_scale=self.max_scale_pct,
            max_rotate=self.max_rotate_degrees,
            max_shear=self.max_shear_degrees,
            max_translate=self.max_translate_pixels)

        # Split the data into train, valid, and test
        self.split_load_data()

    def split_load_data(self):
        """ This method splits the data into a list of training, validation, and testing images """
        # Create a list of images we want to learn from
        images = os.listdir(self.path_to_data)
        images = [im for im in images if im.endswith(".npy")]

        # Shuffle the list
        random.shuffle(images)
        n_images = len(images)
        if n_images == 0:
            raise ValueError("No images loaded")

        # Get the number of training and validation images for each set
        n_train = int(np.floor(n_images * self.train_valid_test_split[0]))
        n_valid = int(np.floor(n_images * self.train_valid_test_split[1]))

        # Get the training, testing, and validation sets
        self.images["train"] = images[0:n_train]
        self.images["valid"] = images[n_train:(n_train + n_valid)]
        self.images["test"] = images[(n_train + n_valid):]

    def setup_model(self):
        """ This method compiles the TensorFlow graphs (neural networks) for training """
        # A mapping from strings to class definitions. This
        # is used to build what the user specified in JSON.
        str_class_map = {
            "conv_2d": Conv2D,
            "separable_conv_2d": SeparableConv2D,
            "bayes_conv_2d": Convolution2DReparameterization,
            "average_pooling_2d": AveragePooling2D,
            "max_pooling_2d": MaxPooling2D,
            "dense": Dense,
            "dropout": Dropout,
            "max_pooling_3d": MaxPooling3D,
            "batch_norm": BatchNormalization,
            "depthwise_conv": DepthwiseConv2D,
        }

        # Generate a batch of images to get sizes
        batch = self.generate_warps(num_warps=1)

        # Get the input images
        nn_inputs = batch["X"]

        # Get the shape of the left input into the NN
        nn_input_a_shape = nn_inputs["A"][0].shape
        left_input = Input(nn_input_a_shape, name="left")

        # Get the shape of the right input into the NN
        nn_input_b_shape = nn_inputs["B"][0].shape
        right_input = Input(nn_input_b_shape, name="right")

        # Create a dictionary of sub networks and names of layers
        self.sub_models = {nm: Sequential(name=nm) for nm in self.sub_graphs.keys()}
        self.layer_names = {nm: [] for nm in self.sub_graphs.keys()}

        # Number of filters per repeat
        fmap = {0: 4}
        for i in range(1, 20):
            fmap[i] = fmap[i - 1] * 4

        # For each sub graph in the ensemble (a component)
        for name, params in self.sub_graphs.items():
            # Repeat the layers N times over
            for i in range(params["repeats"]):
                # For each layer in this repeat
                for j, layer in enumerate(params["layers"]):

                    # Name this layer of the sub network
                    x = str(len(self.layer_names[name]))
                    layer_name = "_".join([name, x])

                    # Get the layer class name
                    layer_copy = layer.copy()
                    layer_class = layer_copy.pop("class")

                    if i == 0:
                        # If first layer, specify the shape
                        layer_copy["input_shape"] = nn_input_a_shape

                    if "l2_regularization" in layer_copy:
                        # Replace the amount of l1 with a regularizer
                        l2_reg = layer_copy.pop("l2_regularization")
                        layer_copy["kernel_regularizer"] = l2(l2_reg)

                    if "filters" in layer_copy:
                        if layer_copy["filters"] == "auto":
                            layer_copy["filters"] = fmap[i]

                    # Add a Keras Layer to the Sequential component
                    layer_obj = str_class_map[layer_class](
                        name=layer_name, **layer_copy)
                    self.sub_models[name].add(layer_obj)
                    self.layer_names[name].append(layer_name)

        # Store each ensembles representations
        # of the left and right images
        representations = []

        def siamese_difference(features):
            """ Absolute difference for Lambda """
            return K.abs(features[0] - features[1])

        for name, network in self.sub_models.items():

            # Get the representation of the left & right images
            left = network(left_input)
            right = network(right_input)

            # Compute global average pooling
            gap_left = GlobalMaxPooling2D(
                name=name + "_left_glob_pool")(left)
            gap_right = GlobalMaxPooling2D(
                name=name + "_right_glob_pool")(right)

            # Then compute the absolute difference
            diff = Lambda(
                siamese_difference,
                name=name + "_diff"
            )([gap_left, gap_right])

            # Add this mapping to the representations
            representations.append(diff)

        if len(representations) < 2:
            # There is only one representation!
            maps = representations[0]
        else:
            # Concatenate all the image representations
            maps = Concatenate(
                name="maps"
            )(representations)

        # Flatten and dropout the absolute differences
        flat = Flatten(name="features")(maps)

        # Add the softmax layer
        probabilities = Dense(
            name="probabilities", units=2,
            activation="softmax")(flat)

        # Create the siamese neural network architecture
        self.model = Model([left_input, right_input],probabilities)

        if self.print_summary:
            # Print out a summary
            self.model.summary()
            for arch, model in self.sub_models.items():
                print("\n", arch.upper())
                model.summary()

        # Create the optimizer
        adam_optimizer = Adam(lr=0.001)

        # Compile the siamese network
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=adam_optimizer,
            metrics=[categorical_accuracy])

    def train_model(self):
        """ This method trains the model in sessions """

        if not self.model:
            # Setup the model if it has not
            # been setup and compiled yet
            self.setup_model()

        # Workout the path to the weights to save
        path_to_saved_weights = os.path.join(
            self.path_to_here, self.name + ".h5")

        # Tracking variables for learning routine
        count_without_improvement = 0
        best_val_loss = np.inf
        count = 0

        self.print("=" * 160)
        while count < self.training_sessions:
            self.print("EPOCH:", str(count), "STALE:",
                       str(count_without_improvement))
            self.print("-" * 160)

            # Generate augmented training and validation data
            train, valid = self.generate_epoch_data(
                include_train=True, include_valid=True)

            # Fit the model on the generated data
            process = self.model.fit(
                x=[train["X"]["A"], train["X"]["B"]],
                y=to_categorical(train["y"]),
                validation_data=[
                    [valid["X"]["A"], valid["X"]["B"]],
                    to_categorical(valid["y"])
                ], batch_size=self.batch_size,
                verbose=self.verbose, shuffle=True,
                epochs=self.epochs_per_session,
                callbacks=[ReduceLROnPlateau(
                    verbose=False, patience=2)])

            # Get the loss on the validation set
            val_loss = process.history["val_loss"][-1]
            if val_loss < best_val_loss:
                # The model has improved, saved the new best weights
                self.model.save_weights(path_to_saved_weights)
                self.print("saved new best model to disk ...")
                count_without_improvement = 0
                best_val_loss = val_loss
            else:
                # Increment the count without improvement
                count_without_improvement += 1
            # Next epoch
            count += 1
            self.print("-" * 160)
        self.print("=" * 160)

    def load_trained_weights(self):
        """ This method loads the trained weights into the model """
        if not self.model:
            # Initialize the graph
            self.setup_model()

        try:
            # Load the weights for the full model (incl. ensembles)
            assert isinstance(self.model, Model)
            self.model.load_weights(os.path.join(
                self.path_to_here, self.name + ".h5"))

            # Now loop through every sub model in the ensemble
            for name in self.sub_models.keys():
                # Check the model is Sequential and, if so, load its weights
                assert isinstance(self.sub_models[name], Sequential)
                sub_model = self.model.get_layer(name=name)
                assert isinstance(sub_model, Sequential)
                self.sub_models[name].set_weights(sub_model.get_weights())

        except Exception as e:
            # Otherwise ask the user if they want to retrain the model
            do_train = input("issue loading weights, train? [y/n]:")
            if do_train.lower() == 'y':
                # Attributes relating to prints
                self.print_summary = True
                self.verbose = True
                self.train_model()

    def evaluate_model(self):
        """ This method performs an ablation study at the ensemble level """
        # Generate some data for evaluation
        train, valid = self.generate_epoch_data(
            include_train=True, include_valid=True)

        # Get the baseline score - with all incl.
        baseline = self.model.evaluate(
            x=list(valid["X"].values()), verbose=1,
            y=to_categorical(list(valid["y"])),
            batch_size=self.batch_size)
        baseline = np.array(baseline)
        zeros = np.zeros(baseline.shape)

        # Create a dictionary to store the score differentials in
        score_differentials = {sg: zeros.copy() for sg in self.layer_names.keys()}

        for sub_graph, layer_names in self.layer_names.items():
            layer_weights = {ln: None for ln in layer_names}
            sub_model = self.sub_models[sub_graph]

            for layer_name in layer_names:
                # Ablate each of the weights in this sub-graph
                layer = sub_model.get_layer(name=layer_name)
                weights = layer.get_weights()
                layer_weights[layer_name] = weights.copy()
                zeros = [np.zeros(w.shape) for w in weights]
                layer.set_weights(zeros)

            # Score the model without this sub graph
            score = self.model.evaluate(
                x=list(valid["X"].values()), verbose=0,
                y=to_categorical(list(valid["y"])),
                batch_size=self.batch_size)

            # Score the differential in the score
            differential = baseline - score
            score_differentials[sub_graph] = differential

            for layer_name in layer_names:
                layer = sub_model.get_layer(name=layer_name)
                layer.set_weights(layer_weights[layer_name])

        # Return the baseline and the diffs
        return baseline, score_differentials

    def prepare_image(self, input_image):
        """ This method just scales the image to between 0. and 1. """
        shape = input_image.shape
        if shape[0] != self.image_size[0] or shape[1] != self.image_size[1]:
            # Resize the images so that they have a consistent shape
            input_image = resize(input_image, self.image_size)
        # Convert the image to a float representation between 0 and 1
        return np.array(input_image, dtype=np.float16) / 255.

    def get_similarity(self, image_a, image_b):
        """
        This method returns the probability the model has assigned to image_a and image_b being the same and the
        inverse probability i.e. the probability the model has assigned to image_a and image_b NOT being the same.
        :param image_a: the left image
        :param image_b: the right image
        :return: P(image_a == image_b) and P(image_a != image_b)
        """
        # Prepare the images for classification
        image_a = self.prepare_image(image_a)
        image_b = self.prepare_image(image_b)
        prob = self.model.predict(
            x=[[image_a], [image_b]]).flatten()
        p_diff, p_same = prob
        return p_same

    def get_features(self, image_a, image_b):
        """
        This method returns the final, final features extracted by the global max pooling layers before they are
        fed into the probability distribution at the end which returns P(same) and P(different).
        :param image_a: the left image
        :param image_b: the right image
        :return: the final features learned by the model
        """
        # Prepare the images for classification
        image_a = self.prepare_image(image_a)
        image_b = self.prepare_image(image_b)
        features = self.model.get_layer('features')
        feature_sub_model = Model(
            inputs=self.model.input,
            outputs=features.output)
        feature_values = feature_sub_model.predict(
            x=[[image_a], [image_b]]).flatten()
        return feature_values

    def plt_features(self, image_a, image_b, file_name=None):
        """
        This method plots the final features extracted from the global max pooling layers.
        :param image_a: the left image
        :param image_b: the right image
        """
        features = self.get_features(image_a=image_a, image_b=image_b)
        x_axis = list(range(len(features)))
        plt.bar(x_axis, features)
        if not file_name:
            plt.show()
        else:
            plt.savefig(file_name + ".png")
            plt.close()

    def get_features_similarity(self, image_a, image_b):
        """
        Returns the similarity based on the global max pooling features.
        :param image_a: the left image
        :param image_b: the right image
        :return: the similarity of the two images
        """
        features = self.get_features(
            image_a=image_a, image_b=image_b)
        return np.nansum(features)

    def get_feature_maps(self, input_image):
        """
        This method returns the final-layer feature maps extracted from each of the feature extractor neural networks
        in the ensemble of neural networks that make up the AugSimNet siamese ensemble.
        :param input_image: the image we want features of
        :return: a dictionary of feature maps
        """
        # Prepare the input image for classification
        input_image = self.prepare_image(input_image)
        # Create a dictionary of feature maps for each sub model
        feature_maps = {name: [] for name in self.sub_models.keys()}
        for name, sub_model in self.sub_models.items():
            # Extract the feature map from the sub model
            feature_map = sub_model.predict([[input_image]])[0]
            h, w, c = list(feature_map.shape)
            for ci in range(c):
                feature_maps[name].append(feature_map[:, :, ci])
        # Return the feature maps
        return feature_maps

    def plt_feature_maps(self, input_image, file_name=None):
        """
        This method plots the feature maps learnt by each of the feature extractor neural networks
        :param input_image: the image we want features of
        """
        mappings = {1: (1, 1), 2: (1, 2), 4: (2, 2), 8: (2, 4),
                    16: (4, 4), 32: (4, 8), 64: (8, 8)}
        # Plot the original image for comparison
        feature_maps = self.get_feature_maps(input_image)
        for name, features in feature_maps.items():
            ps1, ps2 = mappings[len(features)]
            # Create the image with an appropriate grid size
            f, ax = plt.subplots(ps1, ps2, figsize=(16, 8))
            f.suptitle(name.upper(), fontsize=16)
            ax = ax.flatten()
            for i, feature in enumerate(features):
                # Plot this extracted feature map
                ax[i].matshow(feature)
                ax[i].axis("off")

            if not file_name:
                plt.show()
            else:
                plt.savefig(file_name + "_" + name + ".png")
                plt.close()

    def plt_all_feature_maps(self, arch_name, input_image, file_name=None):
        """
        This method plots the feature maps produced by each successive layer of a specified feature extactor neural
        network. This is useful for understanding what kinds of features different architectures have learnt.
        :param arch_name: the name of the feature extractor neural network to visualize
        :param input_image: the image we want to visualize the feature extraction for
        """
        input_image = self.prepare_image(input_image)

        # Get the model we want features for
        model = self.sub_models[arch_name]
        assert isinstance(model, Model)

        mappings = {1: (1, 1), 2: (1, 2), 4: (2, 2), 8: (2, 4),
                    16: (4, 4), 32: (4, 8), 64: (8, 8)}

        for depth, layer in enumerate(model.layers):

            function = Model(model.input, layer.output)
            feature_map = function.predict([[input_image]])[0]
            h, w, c = list(feature_map.shape)

            ps1, ps2 = mappings[c]
            f, ax = plt.subplots(
                ps1, ps2, figsize=(16, 8))

            if isinstance(ax, np.ndarray):
                ax = ax.flatten()
                for ci in range(c):
                    # Plot the extracted feature
                    ax[ci].imshow(feature_map[:, :, ci])
                    ax[ci].axis("off")
            else:
                # There's only one feature
                ax.imshow(feature_map[:, :, 0])

            f.suptitle("FEATURE MAPS AT DEPTH " + str(depth))

            if not file_name:
                plt.show()
            else:
                plt.savefig(file_name + "_depth=" + str(depth) + ".png")
                plt.close()

    def get_feature_map_similarity(self, image_a, image_b):
        """
        This method returns the mean squared distance between the final features learnt by each of the feature
        extraction neural networks for image A and image B. This metric is used to align the images.
        :param image_a: the left image
        :param image_b: the right image
        :return: the mean squared error between left and right
        """
        all_image_a_features = self.get_feature_maps(image_a)
        all_image_b_features = self.get_feature_maps(image_b)
        cumulative_error = 0.
        for name in all_image_a_features.keys():
            image_a_features = all_image_a_features[name]
            image_b_features = all_image_b_features[name]
            n = len(image_a_features)
            name_error = 0.
            for i in range(n):
                # Get the feature maps and compute the difference
                a_map = image_a_features[i].flatten()
                b_map = image_b_features[i].flatten()
                name_error += np.sum(np.square(a_map - b_map))
            # Add this feature extractors errors
            cumulative_error += name_error
        return cumulative_error

    def generate_epoch_data(self, include_train=True, include_valid=True):
        """
        This method produces training patterns and responses for an epoch.
        :param include_train: include pattern data, True or False?
        :param include_valid: include response data, True of False
        :return: a dictionary containing patterns (X) and responses (y)
        """
        training_data = {"X": {"A": [], "B": []}, "y": []}
        validation_data = {"X": {"A": [], "B": []}, "y": []}

        num_images = max(
            self.images_per_evaluation,
            self.images_per_training_session)

        bar = tqdm.tqdm(
            iterable=range(num_images), desc="GENERATING DATA",
            ncols=160, ascii=True, mininterval=0.0001)

        for i in bar:

            if include_train and i < self.images_per_training_session:
                train_i = self.generate_warps(
                    num_warps=self.warps_per_image, mode="train")
                for k, v in train_i["X"].items():
                    training_data["X"][k].extend(v)
                training_data["y"].extend(train_i["y"])

            if include_valid and i < self.images_per_evaluation:
                valid_i = self.generate_warps(
                    num_warps=self.warps_per_image, mode="valid")
                for k, v in valid_i["X"].items():
                    validation_data["X"][k].extend(v)
                validation_data["y"].extend(valid_i["y"])

        # Garage collector
        # for good luck
        gc.collect()
        return training_data, validation_data

    def generate_warps(self, num_warps=4, mode="train"):
        """
        This method with randomly pick two images from the training, validation, or test set and produce N warped
        versions of those images as a batch. This is used to generate the training data for the network.
        :param num_warps: the number of warps to produce.
        :param mode: the set to sample the underlying image from.
        """
        num_warps = max(int(num_warps / 4), 1)

        # Load a pair of image - same and not same
        image_a_name = random.choice(self.images[mode])
        image_b_name = random.choice(self.images[mode])

        # Ensure the images are different
        while image_a_name == image_b_name:
            image_b_name = random.choice(self.images[mode])

        batch = {"X": {"A": [], "B": []}, "y": []}
        for i in range(num_warps):
            a_1, b_1, a_aug_1, b_aug_1, trans_a_1, trans_b_1 = self.generate_warp(
                image_a_name=image_a_name, image_b_name=image_b_name)
            a_2, b_2, a_aug_2, b_aug_2, trans_a_2, trans_b_2 = self.generate_warp(
                image_a_name=image_a_name, image_b_name=image_b_name)

            a_aug_1 = self.prepare_image(a_aug_1)
            a_aug_2 = self.prepare_image(a_aug_2)

            same, diff = 1, 0

            # Add the positive sample
            batch["X"]["A"].append(a_aug_1)
            batch["X"]["B"].append(a_aug_2)
            batch["y"].append(same)

            # Add the positive sample
            batch["X"]["A"].append(b_aug_1)
            batch["X"]["B"].append(b_aug_2)
            batch["y"].append(same)

            # Add the negative sample
            batch["X"]["A"].append(a_aug_1)
            batch["X"]["B"].append(b_aug_1)
            batch["y"].append(diff)

            # Add the negative sample
            batch["X"]["A"].append(a_aug_2)
            batch["X"]["B"].append(b_aug_2)
            batch["y"].append(diff)

        return batch

    def generate_warp(self, image_a_name: str, image_b_name: str):
        """ This method returns the data required for the training patterns """
        image_a = self.load_image(image_name=image_a_name)
        image_b = self.load_image(image_name=image_b_name)
        image_a, image_a_augmented, transform_a = self.warp_image(image=image_a)
        image_b, image_b_augmented, transform_b = self.warp_image(image=image_b)
        return image_a, image_b, image_a_augmented, image_b_augmented, transform_a, transform_b

    def load_image(self, image_name):
        """ Loads an image from disk - nothing special """
        return np.load(os.path.join(self.path_to_data, image_name))

    def warp_image(self, image, method="uniform"):
        """ Applies a random move vector to the given image """
        valid, tries = False, 0
        while not valid:
            image = resize(image, self.image_size)
            move_vector = self.image_mover.sample_move_vector(method=method)
            moved = self.image_mover.move_image(
                image_in=image, move_vector=move_vector)
            valid, tries = np.sum(moved) > 0, tries + 1
            if tries > 5:
                # Give up and just return the original image
                return image, image, np.array([1., 1., 0., 0., 0, 0])
        # Return the moved image and vector
        return image, moved, move_vector

    def reinit_model(self):
        """ Helper method for reinitializing the model """
        if not self.model:
            # No need to reinit just init
            self.setup_model()
        else:
            # Reinitialize every layer with an initialized
            session = K.get_session()
            for layer in self.model.layers:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.initializer.run(session=session)

    def print(self, *args):
        """ Helper method for printing to console """
        if self.verbose:
            print(*args)
