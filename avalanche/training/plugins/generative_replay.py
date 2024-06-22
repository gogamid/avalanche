from copy import deepcopy
from typing import Optional, Any
from avalanche.core import SupervisedPlugin, Template
import torch


class GenerativeReplayPlugin(SupervisedPlugin):
    """
    Generative Replay (GR) plugin.
    Updates the current mbatch of a strategy before training an experience
    by sampling a generator model and concatenating the replay data to the
    current batch.
    There are three ways to implement generative replay. The first is to
    generate constant replay size data and append to each minibatch before
    iteration. Another way is to increase the replay size where with each
    iteration train_mb_size times current experience index will be added
    to each minibatch before each iteration. The third way is by weighting
    the loss function and give more importance to the replayed data as the
    number of experiences increases. In this way train_mb_size samples are
    generated and the loss is computed as a weighted sum of the data loss
    and the replay loss with adaptive ratio.
    """

    def __init__(
        self,
        generator_strategy=None,
        untrained_solver: bool = True,
        replay_size: Optional[int] = None,
        increasing_replay_size: bool = False,
    ):
        """
        :param generator_strategy: In case the plugin is applied to a non-generative
            model (e.g. a simple classifier), this should contain an Avalanche strategy
            for a model that implements a 'generate' method
            (see avalanche.models.generator.Generator). Defaults to None.
        :param untrained_solver: if True we assume this is the beginning of
            a continual learning task and add replay data only from the second
            experience onwards, otherwise we sample and add generative replay data
            before training the first experience. Default to True.
        :param replay_size: The user can specify the batch size of replays that
            should be added to each data batch. By default each data batch will be
            matched with replays of the same number.
        :param increasing_replay_size: If True, in each iteration train_mb_size
            times current experience index will be added to each minibatch
            before each iteration.
        """
        super().__init__()
        self.generator_strategy = generator_strategy
        if self.generator_strategy:
            self.generator = generator_strategy.model
        else:
            self.generator = None
        self.untrained_solver = untrained_solver
        self.model_is_generator = False
        self.replay_size = replay_size
        self.increasing_replay_size = increasing_replay_size

    def before_training(self, strategy, *args, **kwargs):
        """Checks whether we are using a user defined external generator
        or we use the strategy's model as the generator.
        If the generator is None after initialization
        we assume that strategy.model is the generator.
        (e.g. this would be the case when training a VAE with
        generative replay)"""
        if not self.generator_strategy:
            self.generator_strategy = strategy
            self.generator = strategy.model
            self.model_is_generator = True

    def before_training_exp(
        self, strategy, num_workers: int = 0, shuffle: bool = True, **kwargs
    ):
        """
        Make deep copies of generator and solver before training new experience.
        """
        if self.untrained_solver:
            # The solver needs to be trained before labelling generated data and
            # the generator needs to be trained before we can sample.
            return
        self.old_generator = deepcopy(self.generator)
        self.old_generator.eval()
        if not self.model_is_generator:
            self.old_model = deepcopy(strategy.model)
            self.old_model.eval()

    def after_training_exp(
        self, strategy, num_workers: int = 0, shuffle: bool = True, **kwargs
    ):
        """
        Set untrained_solver boolean to False after (the first) experience,
        in order to start training with replay data from the second experience.
        """
        self.untrained_solver = False

    def calculate_replay_size(self, strategy):
        if self.replay_size:
            return self.replay_size
        else:
            if self.increasing_replay_size:
                return strategy.train_mb_size * strategy.experience.current_experience
            else:
                return strategy.train_mb_size

    def before_training_iteration(self, strategy, **kwargs):
        """
        Generating and appending replay data to current minibatch before
        each training iteration.
        """
        if self.untrained_solver:
            return

        replay_size = self.calculate_replay_size(strategy)

        # extend inputs
        replay = self.old_generator.generate(replay_size).to(strategy.device)
        strategy.mbatch[0] = torch.cat([strategy.mbatch[0], replay], dim=0)

        # extend labels
        if not self.model_is_generator:
            with torch.no_grad():
                replay_output = self.old_model(replay).argmax(dim=-1)
        else:
            replay_output = torch.zeros(replay.shape[0])

        strategy.mbatch[1] = torch.cat(
            [strategy.mbatch[1], replay_output.to(strategy.device)], dim=0
        )

        # extend task ids(we implicitley assume a task-free case)
        strategy.mbatch[-1] = torch.cat(
            [
                strategy.mbatch[-1],
                torch.ones(replay.shape[0]).to(strategy.device)
                * strategy.mbatch[-1][0],
            ],
            dim=0,
        )


class TrainGeneratorAfterExpPlugin(SupervisedPlugin):
    """
    TrainGeneratorAfterExpPlugin makes sure that after each experience of
    training the solver of a scholar model, we also train the generator on the
    data of the current experience.
    """

    def after_training_exp(self, strategy, **kwargs):
        """
        The training method expects an Experience object
        with a 'dataset' parameter.
        """
        for plugin in strategy.plugins:
            if type(plugin) is GenerativeReplayPlugin:
                print("Training generator after experience")
                plugin.generator_strategy.train(strategy.experience)


__all__ = ["GenerativeReplayPlugin", "TrainGeneratorAfterExpPlugin"]
