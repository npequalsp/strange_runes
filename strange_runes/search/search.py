from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from strange_runes.image_tools import ImageMover
from strange_runes.similarity.siamese import AugSimNet

np.set_printoptions(suppress=True)


class ImageSearch(metaclass=ABCMeta):
    """
    The ImageSearch base class exposes methods for easily constructing search algorithms to do image registration.

    Each ImageSearch object is given:

    * A floating image (starting unaligned image)
    * A fixed image (the image we want to align to)
    * Parameters relating to the allowable transformations
    * An AugSimNet object for measuring similarity

    The goal of the search algorithm is then to find the optimal set of transformation parameters (called move vectors
    in the code) that minimize the distance between the floating image after transformation and the fixed image.

    The key to solving this problem is constructing a robust measure of similarity. AugSimNet, an augmented siamese
    neural network trained on warped images, exposes 3 methods for measuring similarity namely:

    1. diff_probability - minimize the probability that the images are different
    2. feature_vector - minimize the sum of the siamese neural networks' logits
    3. feature_maps - minimize the sum squared error between the siamese NN's feature maps

    In addition to the above a fourth option, raw_image, is provided which does not rely on the AugSimNet object. This
    was added for comparative purposes. The ability of the siamese neural network to extract relevant features and
    them for the search algorithm to align based on those features is the most robust approach exposed.
    """

    def __init__(self, fixed_image_goal, floating_image_start, similarity_net, *args, **kwargs):
        """
        Constructs an abstract Search object for aliging images
        :param fixed_image_goal: the image we want to align to
        :param floating_image_start: the image we want to align
        :param similarity_net: an AugSimNet object for measuring similarity
        """
        # Store the fixed image - the goal
        self.fixed_image_goal = fixed_image_goal
        # Store the floating image - the starting point
        self.floating_image_start = floating_image_start

        # An AugSimNet object for measuring image similarity
        self.similarity_net = similarity_net
        self.similarity_metric = "feature_maps"

        # Record the image size / shape
        self.image_size = (512, 512)

        self.max_scale_pct = 0.15
        self.max_rotate_degrees = 365
        self.max_translate_pixels = 128
        self.max_shear_degrees = 5

        for kw, attr in kwargs.items():
            setattr(self, kw, attr)

        # Memoization for score func
        self.score_memo = {}

        # An ImageMover for manipulating images
        self.image_mover = ImageMover(
            height=self.image_size[0],
            width=self.image_size[1],
            max_scale=self.max_scale_pct,
            max_rotate=self.max_rotate_degrees,
            max_shear=self.max_shear_degrees,
            max_translate=self.max_translate_pixels)

        # Double check that the correct types are passed
        assert isinstance(self.similarity_net, AugSimNet)
        assert isinstance(self.image_mover, ImageMover)

    @abstractmethod
    def search(self, *args, **kwargs):
        """ This is the abstract method for the search logic """
        raise NotImplementedError("search() not implemented")

    def score(self, move_vector):
        """ Returns the fitness of the move vector """
        # Fix the move vector - round and convert to ints
        move_vector = self.image_mover.fix_move_vector(move_vector)
        key = "-".join([str(mv) for mv in move_vector])

        if key in self.score_memo:
            # Return previously computed score
            return self.score_memo[key]
        else:
            # First compute the penalty term for violations
            penalty = self.image_mover.penalize_move_vector(move_vector)

            if self.similarity_metric == "diff_probability":
                # Return probability that the images are different
                moved_image = self.move_image(move_vector=move_vector)
                score = 1. - self.similarity_net.get_similarity(
                    image_a=self.fixed_image_goal, image_b=moved_image) + penalty

            elif self.similarity_metric == "feature_vector":
                # Return the MSE computed from the final feature vector
                moved_image = self.move_image(move_vector=move_vector)
                score = self.similarity_net.get_features_similarity(
                    image_a=self.fixed_image_goal, image_b=moved_image) + penalty

            elif self.similarity_metric == "feature_maps":
                # Return the MSE computed from the feature maps
                moved_image = self.move_image(move_vector=move_vector)
                score = self.similarity_net.get_feature_map_similarity(
                    image_a=self.fixed_image_goal, image_b=moved_image) + penalty

            elif self.similarity_metric == "raw_image":
                # Return the MSE computed from the feature maps
                moved_image = self.move_image(move_vector=move_vector)
                score = np.nansum(np.square(self.fixed_image_goal - moved_image))

            else:
                # Raise an attribute error in this case.
                raise AttributeError("Invalid similarity metric.")
            # Store the score in the memoizer
            self.score_memo[key] = score
            return score

    def plt_result(self, move_vector, name=None):
        """ This method plots the results of the search algorithm. """
        image_optimal = self.move_image(move_vector=move_vector)

        f, ax = plt.subplots(1, 3, figsize=(16, 8))
        f.suptitle(move_vector)

        ax[0].imshow(self.fixed_image_goal)
        ax[0].set_title("FIXED IMAGE")
        ax[0].set_xticks(np.arange(0, 512, 32), minor=False)
        ax[0].set_yticks(np.arange(0, 512, 32), minor=False)
        ax[0].grid(color='k', linestyle='-', linewidth=1, alpha=0.5)
        ax[0].tick_params(top='on', bottom='on', left='on', right='on',
                          labelleft='off', labelbottom='off')

        ax[1].imshow(self.floating_image_start)
        ax[1].set_title("FLOATING IMAGE")
        ax[1].set_xticks(np.arange(0, 512, 32), minor=False)
        ax[1].set_yticks(np.arange(0, 512, 32), minor=False)
        ax[1].grid(color='k', linestyle='-', linewidth=1, alpha=0.5)
        ax[1].tick_params(top='on', bottom='on', left='on', right='on',
                          labelleft='off', labelbottom='off')

        ax[2].imshow(image_optimal)
        ax[2].set_title("ALIGNED IMAGE")
        ax[2].set_xticks(np.arange(0, 512, 32), minor=False)
        ax[2].set_yticks(np.arange(0, 512, 32), minor=False)
        ax[2].grid(color='k', linestyle='-', linewidth=1, alpha=0.5)
        ax[2].tick_params(top='on', bottom='on', left='on', right='on',
                          labelleft='off', labelbottom='off')

        if name:
            plt.savefig(name)
            plt.close()
        else:
            plt.show()

    def sample_move_vector(self, method="uniform"):
        """ Wrapper to ImageMove.sample_move_vector """
        return self.image_mover.sample_move_vector(method=method)

    def sample_move_vector_set(self, set_size, method="uniform"):
        """ Wrapper to ImageMover.sample_move_vector_set """
        return self.image_mover.sample_move_vector_set(set_size=set_size, method=method)

    def move_image(self, move_vector):
        """ Wrapper to ImageMover.move_image applied only to the floating image"""
        return self.image_mover.move_image(image_in=self.floating_image_start, move_vector=move_vector)

    def fix_move_vector(self, move_vector):
        """ Wrapper to ImageMover.fix_move_vector """
        return self.image_mover.fix_move_vector(move_vector=move_vector)
