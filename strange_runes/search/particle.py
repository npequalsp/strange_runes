import os

import numpy as np
import tqdm

from strange_runes.search.search import ImageSearch
from strange_runes.search.greedy import GreedyImageSearch
from strange_runes.similarity.siamese import AugSimNet
import numpy.random as rng


class ParticleImageSearch(ImageSearch):
    def search(self, num_epochs=100, swarm_size=20, w=0.729844, social=1.49618,
               cognitive=1.49618, method="normal", greedy=True, show_bar=True):
        """
        Implements a particle swarm search algorithm.
        :param num_epochs: the number of epochs to continue for.
        :param swarm_size: the size of the swarm (number of solutions)
        :param w: the dampening factor to apply to the velocities
        :param social: the weight to apply to the social component
        :param cognitive: the weight to apply to the cognitive component
        :param method: the method for sampling the swarm
        :param greedy: whether to apply greedy search at the end
        :param show_bar: show a progress bar, true or false
        :return: the optimal transformation found and its score
        """
        # Initialize the swarm and their velocities
        swarm = self.sample_move_vector_set(set_size=swarm_size, method=method)
        swarm_velocities = [rng.uniform(-0.01, 0.01, 6) for _ in range(swarm_size)]
        swarm_p_bests_scores, swarm_p_bests = np.array([np.inf for _ in range(swarm_size)]), swarm.copy()

        if show_bar:
            # Create a nice progress bar
            progress = tqdm.tqdm(
                iterable=range(int(num_epochs)), ascii=True,
                desc="initializing ...", ncols=120)
        else:
            # Don't print any progress bar
            progress = range(int(num_epochs))

        curr_g_best_index = 0
        best_score = np.inf
        for _ in progress:

            best_similarity = np.inf
            swarm_similarity = 0.

            for j in range(swarm_size):
                particle_j_similarity = self.score(swarm[j])
                swarm_similarity += particle_j_similarity

                if particle_j_similarity < swarm_p_bests_scores[j]:
                    swarm_p_bests_scores[j] = particle_j_similarity
                    swarm_p_bests[j] = swarm[j].copy()

                if particle_j_similarity < best_similarity:
                    best_similarity, curr_g_best_index = particle_j_similarity, j

            for j in range(swarm_size):
                if j != curr_g_best_index:
                    r1, r2 = rng.uniform(0., 1., 6), rng.uniform(0., 1., 6)
                    social_update = social * r1 * (np.array(swarm[curr_g_best_index]) - np.array(swarm[j]))
                    cognitive_update = cognitive * r2 * (np.array(swarm_p_bests[j]) - np.array(swarm[j]))
                    swarm_velocities[j] = (w * swarm_velocities[j]) + social_update + cognitive_update
                    swarm[j] = np.array(swarm[j]) + swarm_velocities[j]

            best_score = self.score(swarm[curr_g_best_index])

            if show_bar:
                swarm_score = swarm_similarity / swarm_size
                progress.set_description(
                    "f(best)=" + str(round(best_score, 3)) + ";" +
                    "f(swarm)=" + str(round(swarm_score, 3)))

        if greedy:
            # Create a local greedy searcher
            greedy_searcher = GreedyImageSearch(
                fixed_image_goal=self.fixed_image_goal,
                floating_image_start=self.floating_image_start,
                similarity_net=self.similarity_net)

            # Perform a greedy search on the best result
            swarm[curr_g_best_index], best_score = greedy_searcher.search(
                start_move_vector=swarm[curr_g_best_index], show_bar=False)

        return self.fix_move_vector(swarm[curr_g_best_index]), best_score
