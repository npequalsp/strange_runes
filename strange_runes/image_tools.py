from random import uniform, randint, normalvariate, shuffle
from skimage.transform import AffineTransform, warp
import numpy.random as rng
import numpy as np
from collections import OrderedDict


class LimitedSizeDict(OrderedDict):
    """
    Implements a LimitedSizeDict - courtesy of the following helpful StackOverflow post:
    https://stackoverflow.com/questions/2437617/how-to-limit-the-size-of-a-dictionary
    """
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


class ImageMover:
    """
    The ImageMover class contains the logic required to manipulate images using affine transformations. Affine
    transformations consist of 4 transformations involving 6 parameters:

    * scale(x, y) - zoom into and out of a picture.
    * rotate(r) - rotate the image counter-clockwise
    * shear(r) - shear the image counter clockwise
    * translate(x, y) - move the image to a new location

    These four transformations are exposed by the ImageMover class. Because these transformations can be
    rather computationally expensive in Python using scikit-image, the ImageMover also includes caching.
    """
    def __init__(self, height=512, width=512, max_scale=0.5, max_rotate=365,
                 max_shear=5, max_translate=256, cache_sizes=1024):
        """
        Constructs an ImageMover object
        :param height: the heights of the images
        :param width: the widths of the images
        :param max_scale: the maximum scaling factor
        :param max_rotate: the maximum rotation in degrees (not radians)
        :param max_shear: the maximum sheer in degrees (not radians)
        :param max_translate: the maximum translation in pixels
        """
        # Record the image size / shape
        self.image_size = (height, width)
        # Specify the max scale as a %
        self.max_scale_pct = max_scale
        # Specify the max rotation as degrees
        self.max_rotate_degrees = max_rotate
        # Specify the max shear as degrees
        self.max_shear_degrees = max_shear
        # Specify the max translation as pixels
        self.max_translate_pixels = max_translate

        # Controls if caching is on / off
        self.do_cache = True

        # Memoization dictionaries for the transformations
        self.scale_image_cache = LimitedSizeDict(size_limit=cache_sizes)
        self.rotate_image_cache = LimitedSizeDict(size_limit=cache_sizes)
        self.shear_image_cache = LimitedSizeDict(size_limit=cache_sizes)
        self.translate_image_cache = LimitedSizeDict(size_limit=cache_sizes)

    def move_image(self, image_in, move_vector):
        """
        This method applies the scale, rotate, shear, and translate transformations independently.
        :param image_in: the image we want to move using the transformations.
        :param move_vector: the parameters of each of the transformations.
        """
        # Create a local copy to avoid issues
        moved_image = image_in.copy()
        # Scale the image by the x and y amounts
        if move_vector[0] != 1. or abs(move_vector[1]) != 1.:
            moved_image = self.scale_image(
                image_in=moved_image,
                x_zoom=move_vector[0],
                y_zoom=move_vector[1])
        # Rotate the image by the given degrees
        if move_vector[2] != 0:
            moved_image = self.rotate_image(
                image_in=moved_image,
                rotate_radians=move_vector[2])
        # Shear the image by the given degrees
        if move_vector[3] != 0.:
            moved_image = self.shear_image(
                image_in=moved_image,
                shear_radians=move_vector[3])
        # Translate the image by the given x and y values
        if move_vector[4] != 0 or move_vector[5] != 0:
            moved_image = self.translate_image(
                image_in=moved_image,
                x_tran=move_vector[4],
                y_tran=move_vector[5])
        # Remove the moved image
        return moved_image

    def sample_move_vector(self, method="normal"):
        """
        This method returns a random move vector
        :param method: either normal or uniform
        """
        if method == "normal":
            # Return a normally distributed move vector
            return self.sample_move_vector_normal()
        elif method == "uniform":
            # Return a uniformly distributed move vector
            return self.sample_move_vector_uniform()
        else:
            # Throw an exception about this invalid method
            raise AttributeError("Invalid method specified")

    def sample_move_vector_uniform(self):
        """ This method returns a move vector sampled uniformly from the allowable region """
        # Generate a uniform random scale between the minimum and maximum
        rand_x_scale = uniform(1. - self.max_scale_pct, 1. + self.max_scale_pct)
        rand_y_scale = uniform(1. - self.max_scale_pct, 1. + self.max_scale_pct)
        # Generate random rotation and shear - measured in radians
        rand_rotate_r = np.deg2rad(uniform(0., self.max_rotate_degrees))
        rand_shear_r = np.deg2rad(uniform(0., self.max_shear_degrees))
        # Generate random x and y translations - measured in # pixels
        rand_x_translate = randint(-self.max_translate_pixels, self.max_translate_pixels)
        rand_y_translate = randint(-self.max_translate_pixels, self.max_translate_pixels)
        # Put together the move vector consisting of these things
        move_vector = [rand_x_scale, rand_y_scale, rand_rotate_r,
                       rand_shear_r, rand_x_translate, rand_y_translate]
        return np.array(self.fix_move_vector(move_vector))

    def sample_move_vector_normal(self):
        """ This method returns a move vector sampled normally from the allowable region """
        # Generate a normally distributed random scale value between the min and max
        rand_x_scale = normalvariate(mu=1., sigma=self.max_scale_pct / 3.5)
        rand_x_scale = min(max(rand_x_scale, 1. - self.max_scale_pct), 1. + self.max_scale_pct)
        rand_y_scale = normalvariate(mu=1., sigma=self.max_scale_pct / 3.5)
        rand_y_scale = min(max(rand_y_scale, 1. - self.max_scale_pct), 1. + self.max_scale_pct)
        # Generate a normally distributed rotation and shear - measured in radians
        rand_rotate_r = np.deg2rad(min(abs(normalvariate(0, self.max_rotate_degrees / 3.5)), self.max_rotate_degrees))
        rand_shear_r = np.deg2rad(min(abs(normalvariate(0, self.max_shear_degrees / 3.5)), self.max_shear_degrees))
        # Generate a normally distributed x and y translations - measured in # pixels
        rand_x_translate = int(normalvariate(0, self.max_translate_pixels / 3.5))
        rand_x_translate = min(max(rand_x_translate, -self.max_translate_pixels), self.max_translate_pixels)
        rand_y_translate = int(normalvariate(0, self.max_translate_pixels / 3.5))
        rand_y_translate = min(max(rand_y_translate, -self.max_translate_pixels), self.max_translate_pixels)
        # Put together the move vector consisting of these things
        move_vector = [rand_x_scale, rand_y_scale, rand_rotate_r,
                       rand_shear_r, rand_x_translate, rand_y_translate]
        return np.array(self.fix_move_vector(move_vector))

    def sample_move_vector_set(self, set_size, method="normal"):
        """
        This method returns a set of move vectors of size `set_size` sampled according to the method specified. If
        method is set to "normal" the move vectors are sampled normally from the allowable region. If method is set
        to "uniform" the move vectors are sampled uniformly from the allowable region. Uniform sampling results in
        many more extreme transformations of the images whereas normal results in more realistic transformations.
        :param set_size: the number of move vectors to sample
        :param method: normal or uniform
        """
        if method == "normal":
            # Return an approximately normally distributed set of move vectors
            return self.sample_move_vector_set_copula(set_size=set_size)
        elif method == "uniform":
            # Return an approximately uniformly distributed set of move vectors
            return self.sample_move_vector_set_uniform(set_size=set_size)
        else:
            # Throw an exception about this invalid method
            raise AttributeError("Invalid method specified")

    def sample_move_vector_set_uniform(self, set_size):
        """ Samples a set of move vectors uniformly from the allowable region """
        # Generate a set of x and y scaling values to sample move values from
        x_scales = np.linspace(1. - self.max_scale_pct, 1. + self.max_scale_pct, set_size).tolist()
        y_scales = np.linspace(1. - self.max_scale_pct, 1. + self.max_scale_pct, set_size).tolist()
        # Generate a set of radians to sample rotation values from
        rotates_d = np.linspace(0, self.max_rotate_degrees, set_size)
        rotates_r = [np.deg2rad(rd) for rd in rotates_d]
        # Generate a set of radians to sample shear values from
        shears_d = np.linspace(0, self.max_shear_degrees, set_size)
        shears_r = [np.deg2rad(sd) for sd in shears_d]
        # Generate a set of pixel values to sample x and y translations from
        x_translates = np.linspace(-self.max_translate_pixels, self.max_translate_pixels, set_size)
        y_translates = np.linspace(-self.max_translate_pixels, self.max_translate_pixels, set_size)
        x_translates = np.array(x_translates, dtype=np.uint8).tolist()
        y_translates = np.array(y_translates, dtype=np.uint8).tolist()

        # Shuffle everything to give them
        # an equal chance of being popped
        shuffle(x_scales)
        shuffle(y_scales)
        shuffle(rotates_r)
        shuffle(x_translates)
        shuffle(y_translates)

        # Initialize the set with the "do nothing" vector (the origin)
        set = [np.array([1.0, 1.0, 0., 0., 0, 0])]
        for _ in range(1, set_size):
            # Pop an x and y scaling value
            rand_x_scale = x_scales.pop()
            rand_y_scale = y_scales.pop()
            # Pop a rotate and shear value
            rand_rotate_r = rotates_r.pop()
            rand_shear_r = shears_r.pop()
            # Pop and x and y translation value
            rand_x_translate = x_translates.pop()
            rand_y_translate = y_translates.pop()
            # Add this move vector to the set of move vectors
            set.append(np.array(self.fix_move_vector(
                [rand_x_scale, rand_y_scale, rand_rotate_r,
                 rand_shear_r, rand_x_translate, rand_y_translate])))

        # Return the moves
        return set

    def sample_move_vector_set_copula(self, set_size):
        """ Samples a set of move vectors normally from the allowable region """
        # Generate a normally distributed set of x and y scaling values
        x_scales = rng.normal(loc=1., scale=self.max_scale_pct / 3.5, size=set_size).tolist()
        y_scales = rng.normal(loc=1., scale=self.max_scale_pct / 3.5, size=set_size).tolist()
        x_scales = [min(max(x, 1. - self.max_scale_pct), 1. + self.max_scale_pct) for x in x_scales]
        y_scales = [min(max(y, 1. - self.max_scale_pct), 1. + self.max_scale_pct) for y in y_scales]
        # Generate a normally distributed set of rotations and shears - measured in radians
        rotates_d = rng.normal(loc=0, scale=self.max_rotate_degrees / 3.5, size=set_size)
        rotates_d = [min(abs(rd), self.max_rotate_degrees) for rd in rotates_d]
        rotates_r = [np.deg2rad(rd) for rd in rotates_d]
        shears_d = rng.normal(loc=0, scale=self.max_shear_degrees / 3.5, size=set_size)
        shears_d = [min(abs(rd), self.max_shear_degrees) for rd in shears_d]
        shears_r = [np.deg2rad(rd) for rd in shears_d]
        # Generate a normally distributed set of x and y translations - measured in pixels
        x_translates = rng.normal(loc=0, scale=self.max_translate_pixels / 3.5, size=set_size).tolist()
        y_translates = rng.normal(loc=0, scale=self.max_translate_pixels / 3.5, size=set_size).tolist()
        x_translates = np.array(x_translates, dtype=int).tolist()
        y_translates = np.array(y_translates, dtype=int).tolist()
        x_translates = [min(max(x, -self.max_translate_pixels), self.max_translate_pixels) for x in x_translates]
        y_translates = [min(max(y, -self.max_translate_pixels), self.max_translate_pixels) for y in y_translates]

        # Initialize the set with the "do nothing" vector (the origin)
        set = [np.array([1.0, 1.0, 0., 0., 0, 0])]
        for _ in range(1, set_size):
            # Pop an x and y scaling value
            rand_x_scale = x_scales.pop()
            rand_y_scale = y_scales.pop()
            # Pop a rotate and shear value
            rand_rotate_r = rotates_r.pop()
            rand_shear_r = shears_r.pop()
            # Pop and x and y translation value
            rand_x_translate = x_translates.pop()
            rand_y_translate = y_translates.pop()
            # Add this move vector to the set of move vectors
            set.append(np.array(self.fix_move_vector(
                [rand_x_scale, rand_y_scale, rand_rotate_r,
                 rand_shear_r, rand_x_translate, rand_y_translate])))

            # Return the moves
        return set

    def fix_move_vector(self, move_vector):
        """
        This method "fixes" the move vector. basically it removes arbitrary degrees of precision and makes sure that
        the resulting parameters are of the right type. This speeds up the search algorithms.
        :param move_vector: the move vector to fix.
        :return: a fixed version of the move vector.
        """
        move_vector = list(move_vector)
        move_vector[0] = round(move_vector[0], 3)
        move_vector[1] = round(move_vector[1], 3)
        move_vector[2] = round(move_vector[2], 3)
        move_vector[3] = round(move_vector[3], 3)
        move_vector[4] = int(move_vector[4])
        move_vector[5] = int(move_vector[5])
        return move_vector

    def penalize_move_vector(self, move_vector):
        """ Penalize the move vector for breaking the rules """
        penalty = 0.
        if move_vector[0] > self.max_scale_pct + 1.:
            # Add the squared percentage error between the move and the max X scale
            penalty += (((move_vector[0] / self.max_scale_pct) - 1.) * 100.) ** 2.
        if move_vector[1] > self.max_scale_pct + 1.:
            # Add the squared percentage error between the move and the max Y scale
            penalty += (((move_vector[1] / self.max_scale_pct) - 1.) * 100.) ** 2.
        if np.rad2deg(move_vector[2]) > self.max_rotate_degrees:
            # Add the squared percentage error between the move and the max rotation
            penalty += (((np.rad2deg(move_vector[2]) / self.max_rotate_degrees) - 1.) * 100.) ** 2.
        if np.rad2deg(move_vector[3]) > self.max_shear_degrees:
            # Add the squared percentage error between the move and the max rotation
            penalty += (((np.rad2deg(move_vector[3]) / self.max_shear_degrees) - 1.) * 100.) ** 2.
        if move_vector[4] > self.max_translate_pixels:
            # Add the squared percentage error between the move and the max X translation
            penalty += (((move_vector[4] / self.max_translate_pixels) - 1.) * 100.) ** 2.
        if move_vector[5] > self.max_translate_pixels:
            # Add the squared percentage error between the move and the max Y translation
            penalty += (((move_vector[5] / self.max_translate_pixels) - 1.) * 100.) ** 2.
        # Return the penalty
        return penalty

    def hash_image(self, image_in, *args):
        """ Returns a hash of the input image """
        params = ";".join([str(p) for p in args])
        return params + ";" + str(hash(image_in.tostring()))

    def scale_image(self, image_in, x_zoom, y_zoom):
        """ Scales the image (zoom in / zoom out) """
        if self.do_cache:
            # Hash the image and the parameters to get a unique string
            image_in_hash = self.hash_image(image_in, x_zoom, y_zoom)
            if image_in_hash not in self.scale_image_cache:
                # Perform the transformation and cache the result
                result = self.scale_image_logic(image_in, x_zoom, y_zoom)
                self.scale_image_cache[image_in_hash] = result
            return self.scale_image_cache[image_in_hash]
        else:
            # Just return the transformation
            return self.scale_image_logic(image_in, x_zoom, y_zoom)

    def scale_image_logic(self, image_in, x_zoom, y_zoom):
        """ Scales the image (zoom in / zoom out) """
        # Compute the image transformation
        transform = AffineTransform(scale=(x_zoom, y_zoom))
        scaled = warp(image_in, transform, order=1, cval=0,
                      preserve_range=True, mode='constant')
        scaled = scaled.astype(np.uint8)
        if np.sum(scaled) == 0.:
            # Fail gracefully
            return image_in
        else:
            # Return the result
            return scaled

    def rotate_image(self, image_in, rotate_radians):
        """ Rotates the image counter-clockwise by radians. """
        if self.do_cache:
            # Hash the image and the parameters to get a unique string
            image_in_hash = self.hash_image(image_in, rotate_radians)
            if image_in_hash not in self.rotate_image_cache:
                # Perform the transformation and cache the result
                result = self.rotate_image_logic(image_in, rotate_radians)
                self.rotate_image_cache[image_in_hash] = result
            return self.rotate_image_cache[image_in_hash]
        else:
            # Just return the transformation
            return self.rotate_image_logic(image_in, rotate_radians)

    def rotate_image_logic(self, image_in, rotate_radians):
        """ Rotates the image counter-clockwise by radians. """
        # Compute the image transformation
        transform = AffineTransform(rotation=rotate_radians)
        scaled = warp(image_in, transform, order=1, cval=0,
                      preserve_range=True, mode='constant')
        scaled = scaled.astype(np.uint8)
        if np.sum(scaled) == 0.:
            # Fail gracefully
            return image_in
        else:
            # Return the result
            return scaled

    def shear_image(self, image_in, shear_radians):
        """ Shears the image counter-clockwise by radians. """
        if self.do_cache:
            # Hash the image and the parameters to get a unique string
            image_in_hash = self.hash_image(image_in, shear_radians)
            if image_in_hash not in self.shear_image_cache:
                # Perform the transformation and cache the result
                result = self.shear_image_logic(image_in, shear_radians)
                self.shear_image_cache[image_in_hash] = result
            return self.shear_image_cache[image_in_hash]
        else:
            # Just return the transformation
            return self.shear_image_logic(image_in, shear_radians)

    def shear_image_logic(self, image_in, shear_radians):
        """ Shears the image counter-clockwise by radians. """
        # Compute the image transformation
        transform = AffineTransform(shear=shear_radians)
        scaled = warp(image_in, transform, order=1, cval=0,
                      preserve_range=True, mode='constant')
        scaled = scaled.astype(np.uint8)
        if np.sum(scaled) == 0.:
            # Fail gracefully
            return image_in
        else:
            # Return the result
            return scaled

    def translate_image(self, image_in, x_tran, y_tran):
        """ Moves the image from one spot to another """
        if self.do_cache:
            # Hash the image and the parameters to get a unique string
            image_in_hash = self.hash_image(image_in, x_tran, y_tran)
            if image_in_hash not in self.translate_image_cache:
                # Perform the transformation and cache the result
                result = self.translate_image_logic(image_in, x_tran, y_tran)
                self.translate_image_cache[image_in_hash] = result
            return self.translate_image_cache[image_in_hash]
        else:
            # Just return the transformation
            return self.translate_image_logic(image_in, x_tran, y_tran)

    def translate_image_logic(self, image_in, x_tran, y_tran):
        """ Moves the image from one spot to another """
        # Compute the image transformation
        transform = AffineTransform(translation=(x_tran, y_tran))
        scaled = warp(image_in, transform, order=1, cval=0,
                      preserve_range=True, mode='constant')
        scaled = scaled.astype(np.uint8)
        if np.sum(scaled) == 0.:
            # Fail gracefully
            return image_in
        else:
            # Return the result
            return scaled
