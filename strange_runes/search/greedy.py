import os

import numpy as np
import tqdm
from random import uniform

from strange_runes.search.search import ImageSearch
from strange_runes.similarity.siamese import AugSimNet
import numpy.random as rng


class GreedyImageSearch(ImageSearch):
    def search(self, start_move_vector=None, patience=1000, step_fracs=None,
               show_bar=True, shotgun_init=True, shotgun_size=10, method="uniform"):
        """
        Implements a greedy search algorithm
        :param start_move_vector: the move vector to start with
        :param patience: the number of steps to go in any direction whilst improving
        :param step_fracs: the size of the steps to try
        :param show_bar: show a progress bar, true or false
        :param shotgun_init: if true, the starting point is picked at random
        :param shotgun_size: the number of random starting points to consider
        :param method: the method of sampling random starting points
        :return: the optimal transformation and its score
        """
        if start_move_vector is None:
            if shotgun_init:
                # Uniformly generate 1000 possible starting vectors
                shotgun = self.sample_move_vector_set(set_size=shotgun_size, method=method)
                scores = [self.score(move_vector) for move_vector in shotgun]
                start_move_vector = shotgun[np.argmin(np.array(scores))]
            else:
                # Just start with the "no move" move vector
                start_move_vector = np.array([1.0, 1.0, 0., 0., 0, 0])

        move_vector = start_move_vector.copy()
        if step_fracs is None:
            step_fracs = [0.5000, 0.2500, 0.1000,
                          0.0200, 0.0100, 0.0050,
                          0.0020, 0.0010, 0.0005]

        if show_bar:
            # Create a nice progress bar
            progress = tqdm.tqdm(
                iterable=step_fracs, ascii=True,
                desc="initializing ...", ncols=120)
        else:
            # Don't print any progress bar
            progress = step_fracs

        best_score = np.inf
        for step_frac in progress:

            # --- TRY X & Y ZOOM ---

            for j in [0, 1]:

                pre_score = self.score(move_vector)
                improving, count = True, 0

                while improving and count < patience:
                    step = self.max_scale_pct * step_frac
                    step = round(step, 3)
                    step = step * uniform(0., 1.)

                    lft = move_vector.copy()
                    lft[j] -= step
                    lft_score = self.score(lft)

                    rgt = move_vector.copy()
                    rgt[j] += step
                    rgt_score = self.score(rgt)

                    if lft_score < rgt_score:
                        if lft_score >= pre_score:
                            break
                        else:
                            move_vector[j] -= step
                            pre_score = lft_score

                    else:
                        if rgt_score >= pre_score:
                            break
                        else:
                            move_vector[j] += step
                            pre_score = rgt_score

                    count += 1

            # --- TRY ROTATE ---

            for j in [2, 3]:
                pre_score = self.score(move_vector)
                improving, count = True, 0

                while improving and count < patience:

                    step = step_frac * self.max_rotate_degrees
                    step = np.deg2rad(step)
                    step = step * uniform(0., 1.)

                    rgt = move_vector.copy()
                    rgt[j] += step
                    rgt_score = self.score(rgt)

                    if rgt_score < pre_score:
                        move_vector[j] += step
                        pre_score = rgt_score
                    else:
                        break

                    count += 1

            # --- TRY X & Y TRANSLATE ---

            for j in [4, 5]:

                pre_score = self.score(move_vector)
                improving, count = True, 0

                while improving and count < patience:
                    step = self.max_translate_pixels * step_frac
                    step = step * uniform(0., 1.)
                    step = max(int(np.ceil(step)), 1)

                    lft = move_vector.copy()
                    lft[j] -= step
                    lft_score = self.score(lft)

                    rgt = move_vector.copy()
                    rgt[j] += step
                    rgt_score = self.score(rgt)

                    if lft_score < rgt_score:
                        if lft_score >= pre_score:
                            break
                        else:
                            move_vector[j] -= step
                            pre_score = lft_score

                    else:
                        if rgt_score >= pre_score:
                            break
                        else:
                            move_vector[j] += step
                            pre_score = rgt_score

                    count += 1

            best_score = self.score(move_vector)

            if show_bar:
                # Add some useful tracking information
                progress.set_description("f(best)=" + str(round(best_score, 2)))

        return self.fix_move_vector(move_vector), best_score
