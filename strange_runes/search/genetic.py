import os

import numpy as np
import tqdm
from random import uniform, choice, randint

from strange_runes.search.search import ImageSearch
from strange_runes.search.greedy import GreedyImageSearch
from strange_runes.similarity.siamese import AugSimNet
import numpy.random as rng


class GeneticImageSearch(ImageSearch):
    def search(self, num_epochs=20, population_size=100, mutation_rate=0.20, greedy=True,
               method="normal", show_bar=True, crossover_method="random_choice"):
        """
        Implements a genetic search algorithm.
        :param num_epochs: the number of epochs to run for
        :param population_size: the size of the population (num solutions)
        :param mutation_rate: the number of individuals to mutate per epoch
        :param greedy: whether or not to apply greedy search at the end
        :param method: the method of sampling the population
        :param show_bar: show a progress bar, true or false
        :param crossover_method: the method of doing crossover
        :return: the optimal transformation found and it's score
        """
        population = self.sample_move_vector_set(
            set_size=population_size, method=method)

        if show_bar:
            # Create a nice progress bar
            progress = tqdm.tqdm(
                iterable=range(int(num_epochs)), ascii=True,
                desc="initializing ...", ncols=120)
        else:
            # Don't print any progress bar
            progress = range(int(num_epochs))

        best_score = np.inf
        curr_fittest = 0
        for _ in progress:

            # Score the population i.e. obtain their "fitness values"
            fitness_values = np.array([self.score(population[j]) for j in range(population_size)])
            cut_off_fitness, curr_fittest = np.median(fitness_values), np.argmin(fitness_values)

            # Determine which individuals survive this epoch
            survived = [j for j in range(population_size) if
                        fitness_values[j] < cut_off_fitness]

            if len(survived) == 0:
                for j in range(population_size):
                    if j != curr_fittest:
                        # Reinitialize everything except the fittest
                        population[j] = self.sample_move_vector(method=method)
                # Move to next epoch
                continue

            for j in range(population_size):
                if j not in survived:
                    if crossover_method == "random_choice":
                        # Sample two parents to use to create the new individual
                        parents = [population[choice(survived)], population[choice(survived)]]
                        population[j] = np.array([parents[randint(0, 1)][k] for k in range(6)])
                    elif crossover_method == "linear_recombination":
                        recombination = rng.uniform(0., 1., 6)
                        # Randomly sample a new child from the linear hyperplane between the parents
                        parents = [population[choice(survived)], population[choice(survived)]]
                        population[j] = (recombination * parents[0]) + ((1. - recombination) * parents[1])
                    else:
                        # Raise an exception about the invalid crossover method
                        raise ValueError("Invalid crossover method specified")

                if j != curr_fittest and uniform(0., 1.) < mutation_rate:
                    # Sample some noise - this will act as our mutation
                    mutation = [self.sample_move_vector(method=method), population[j]]
                    population[j] = np.array([mutation[randint(0, 1)][k] for k in range(6)])

            best_score = self.score(population[curr_fittest])

            if show_bar:
                population_score = np.mean(fitness_values)
                progress.set_description(
                    "f(best)=" + str(round(best_score, 3)) + ";" +
                    "f(pop)=" + str(round(population_score, 3)))

        if greedy:
            # Create a local greedy searcher
            greedy_searcher = GreedyImageSearch(
                fixed_image_goal=self.fixed_image_goal,
                floating_image_start=self.floating_image_start,
                similarity_net=self.similarity_net)

            # Perform a greedy search on the best result
            population[curr_fittest], best_score = greedy_searcher.search(
                start_move_vector=population[curr_fittest], show_bar=False)

        return self.fix_move_vector(population[curr_fittest]), best_score
