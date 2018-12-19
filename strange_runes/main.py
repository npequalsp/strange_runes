from strange_runes.similarity.siamese import AugSimNet
from strange_runes.search.greedy import GreedyImageSearch
from strange_runes.search.genetic import GeneticImageSearch
from strange_runes.search.particle import ParticleImageSearch

import matplotlib.pyplot as plt
import cProfile
import warnings
import os


def run_through():
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


if __name__ == "__main__":
    run_through()
    # pr = cProfile.run('run_through()')
