from enum import Enum

class Task(Enum):
    """
    Represents the type of task for the model.

    Attributes:
        GRAPH_CLASSIFICATION (int): Task involving classifying entire graphs.
        NODE_CLASSIFICATION (int): Task involving classifying individual nodes within a graph.
        LINK_PREDICTION (int): Task involving predicting the existence or type of edges between nodes.
    """
    GRAPH_CLASSIFICATION = 1
    NODE_CLASSIFICATION = 2
    LINK_PREDICTION = 3

class Stage(Enum):
    """
    Represents the stage of the model lifecycle.

    Attributes:
        TRAINING (int): Indicates the training stage of the model.
        VALIDATION (int): Indicates the validation stage, typically used for hyperparameter tuning.
        TESTING (int): Indicates the testing stage, used for final evaluation of the model.
    """
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3

class Experiment(Enum):
    """
    Represents the type of experiment or variation being conducted.

    Attributes:
        DEFAULT (int): The standard experimental setup.
        GREEDY (int): An experiment using a greedy algorithm for exploration.
        NO_Q (int): An experiment excluding Q-value-based exploration.
        RANDOM (int): An experiment with completely random exploration.
    """
    DEFAULT = 1
    GREEDY = 2
    NO_Q = 3
    RANDOM = 4