# Load Library
import os
import numpy as np
import matplotlib.pyplot as plt
from siamese_network import SiameseNetwork


def show_distributions(data_dir):
    characters_count = []
    for lang in os.listdir(data_dir):
        path = os.path.join(data_dir,lang)
        count = len(os.listdir(path))
        characters_count.append(count)

    # Plot the distribution
    plt.figure(figsize=(20,6))
    plt.bar(os.listdir(data_dir),characters_count)
    plt.xlabel("Languages")
    plt.ylabel("No of characters")
    plt.show()
    
def show_all_characters(data_dir):
    language_list = os.listdir(data_dir)
    selected_lang = np.random.choice(language_list)
    lang_path = os.path.join(data_dir,selected_lang)
    no_of_characters = len(os.listdir(lang_path))
    grid_size = int(np.ceil(np.sqrt(no_of_characters)))
    fig=plt.figure(figsize=(6, 6))
    index = 0
    plt.title("Language:" + selected_lang)
    for characters in os.listdir(lang_path):
        index = index + 1
        char_path = os.path.join(lang_path,characters)
        sample = np.random.choice(os.listdir(char_path))
        sample_path = os.path.join(char_path,sample)
        image = plt.imread(sample_path)
        fig.add_subplot(grid_size,grid_size,index)
        plt.imshow(image)
    plt.show()
    return selected_lang

def show_all_samples(data_dir,language):
    lang_path = os.path.join(data_dir,language)
    selected_character = np.random.choice(os.listdir(lang_path))
    char_path = os.path.join(lang_path,selected_character)
    fig=plt.figure(figsize=(6, 6))
    plt.title("Random characer from " + language)
    index = 0
    for sample in os.listdir(char_path):
        index = index + 1
        sample_path = os.path.join(char_path,sample)
        image = plt.imread(sample_path)
        fig.add_subplot(5,4,index)
        plt.imshow(image)
    plt.show()
    
def show_evaluation_result():
    siamese_network = load_siamese_model()
    evaluation_accuracy = siamese_network.omniglot_loader.one_shot_test(siamese_network.model,20, 40, False)
    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))
    
def load_siamese_model():
    
    dataset_path = 'Omniglot Dataset'
    use_augmentation = True
    learning_rate = 10e-4
    batch_size = 32
    # Learning Rate multipliers for each layer
    learning_rate_multipliers = {}
    learning_rate_multipliers['Conv1'] = 1
    learning_rate_multipliers['Conv2'] = 1
    learning_rate_multipliers['Conv3'] = 1
    learning_rate_multipliers['Conv4'] = 1
    learning_rate_multipliers['Dense1'] = 1
    # l2-regularization penalization for each layer
    l2_penalization = {}
    l2_penalization['Conv1'] = 1e-2
    l2_penalization['Conv2'] = 1e-2
    l2_penalization['Conv3'] = 1e-2
    l2_penalization['Conv4'] = 1e-2
    l2_penalization['Dense1'] = 1e-4
    # Path where the logs will be saved
    tensorboard_log_path = './logs/siamese_net_lr10e-4'
    siamese_network = SiameseNetwork(
        dataset_path=dataset_path,
        learning_rate=learning_rate,
        batch_size=batch_size, use_augmentation=use_augmentation,
        learning_rate_multipliers=learning_rate_multipliers,
        l2_regularization_penalization=l2_penalization,
        tensorboard_log_path=tensorboard_log_path
    )
    # Final layer-wise momentum (mu_j in the paper)
    momentum = 0.9
    # linear epoch slope evolution
    momentum_slope = 0.01
    support_set_size = 20
    evaluate_each = 1000
    number_of_train_iterations = 0

    validation_accuracy = siamese_network.train_siamese_network(number_of_iterations=number_of_train_iterations,
                                                                support_set_size=support_set_size,
                                                                final_momentum=momentum,
                                                                momentum_slope=momentum_slope,
                                                                evaluate_each=evaluate_each, 
                                                                model_name='siamese_net_lr10e-4')

    siamese_network.model.load_weights('./models/siamese_net_lr10e-4.h5')
    return siamese_network
