import os

# Fix OneDNN random issue
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging

# Fix Pymem logging
logging.disable(logging.CRITICAL)

import re
import time
from cmath import e
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gymnasium import spaces

# from logs.PlotLog import PlotLog
from rl.agents import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from CupheadEnv import CupheadEnv
from src.Control import handle_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Config = {
    "RunNumber": 100000,
    "CheckpointInterval": 10000,
    "InitialLR": 0.001,
    "WarmupSteps": 15000,
    "DecaySteps": 100000,
    "DecayRate": 0.1,
    "LogDir": "logs",
    "StartFromCheckpoint": True,
    "CheckpointDir": "checkpoints",
    "CheckpointPrefix": "RewardV2",
    "LoadCheckpointName": "RewardV250000",
}


# Define initial model and layers
def build_model(states, actions):
    model = Sequential(
        [
            keras.layers.Flatten(input_shape=(4, 5)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(
                int(np.prod(actions)),
                activation="linear",
                kernel_regularizer=keras.regularizers.l2(0.01),
            ),
        ]
    )
    return model


# Define agent policy and memory
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.055,
        nb_steps=Config["RunNumber"],
    )
    memory = SequentialMemory(limit=50000, window_length=4)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=Config["WarmupSteps"],
        target_model_update=0.001,
        delta_clip=1.0,
    )
    return dqn


def setup_checkpointing(dqn):
    os.makedirs(Config["CheckpointDir"], exist_ok=True)
    checkpoint_path = os.path.join(
        Config["CheckpointDir"], Config["CheckpointPrefix"] + "{step}.h5f"
    )
    checkpoint_callback = ModelIntervalCheckpoint(
        checkpoint_path, interval=Config["CheckpointInterval"]
    )
    return checkpoint_callback


def setup_file_logger():
    log_dir = Config["LogDir"]
    log_file = os.path.join(log_dir, "training_log.json")

    os.makedirs(log_dir, exist_ok=True)
    file_logger = FileLogger(log_file, interval=100)
    return file_logger


def evaluate_agent(dqn, env, nb_episodes=10, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Evaluative agent for {nb_episodes} episodes.")
    results = dqn.test(env, nb_episodes=nb_episodes, visualize=False, verbose=2)
    avg_reward = np.mean(results.history["episode_reward"])
    logger.info(f"Average reward over {nb_episodes} episodes: {avg_reward}")
    return results


def main(testing=False):
    # Wait for user to configure windows
    time.sleep(3)

    logger.info("Starting training...")

    env = CupheadEnv()
    states = env.observation_space.shape
    actions = env.action_space.n

    # Initialize model
    model = build_model(states, actions)
    model.summary()

    # Initialize agent
    dqn = build_agent(model, actions)

    # Setup checkpointing
    checkpoint_callback = setup_checkpointing(dqn)
    file_logger = setup_file_logger()
    checkpoint_path = os.path.join(
        Config["CheckpointDir"],
        f"{Config['CheckpointPrefix']}{Config['RunNumber']}.h5f",
    )

    # Define learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=Config["InitialLR"],
        decay_steps=Config["DecaySteps"],
        decay_rate=Config["DecayRate"],
    )

    # Define optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    dqn.compile(optimizer=optimizer, metrics=["accuracy"])

    # Initialize level
    handle_action.HandleAction.WorldStartLevel()

    # dqn.model.save("cuphead_model.h5")

    # Determine last checkpoint
    last_checkpoint_num = 0
    if Config["StartFromCheckpoint"]:
        last_checkpoint_num = Config["RunNumber"]
        load_path = os.path.join(
            Config["CheckpointDir"], f"{Config['LoadCheckpointName']}.h5f"
        )
        dqn.load_weights(load_path)
        print(f"Loading checkpoint from {load_path}")

    if not testing:
        # Initialize Plot
        # plot = PlotLog()
        # Learning
        dqn.fit(
            env,
            nb_steps=Config["RunNumber"],
            visualize=False,
            verbose=2,
            callbacks=[checkpoint_callback, file_logger],
        )
        dqn.save_weights(checkpoint_path, overwrite=True)
        logger.info("Training complete.")
        return

    # Testing
    evaluate_agent(dqn, env, nb_episodes=10, logger=logger)
    logger.info("Training complete.")


if __name__ == "__main__":
    main(testing=True)
