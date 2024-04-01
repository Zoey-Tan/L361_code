"""Flower server accounting for Weights&Biases+file saving."""

import timeit
from collections.abc import Callable
from logging import INFO

from flwr.common import Parameters
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy

from project.types.common import ServerRNG
from flwr.common import parameters_to_ndarrays
import numpy as np
from numpy import linalg as LA
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


def calculate_l2_distance(
    ndarray1: list,
    ndarray2: list,
) -> float:
    flatten = []
    for i in range(len(ndarray1)):
        flatten = flatten + (ndarray1[i] - ndarray2[i]).flatten().tolist()
    flatten = np.array(flatten)
    print(flatten.shape)
    return LA.norm(flatten, 2)


def calculate_cosine_similarity(
    ndarray1: list,
    ndarray2: list,
) -> float:
    flatten1 = []
    flatten2 = []
    for ndarray in ndarray1:
        flatten1 = flatten1 + ndarray.flatten()
    for ndarray in ndarray2:
        flatten2 = flatten2 + ndarray.flatten()
    flatten1 = np.array(flatten1)
    flatten2 = np.array(flatten2)


def calculate_pseudo_gradient(
    ndarray1: list,
    ndarray2: list,
) -> list:
    flatten = []
    for i in range(len(ndarray1)):
        flatten = flatten + (ndarray1[i] - ndarray2[i]).flatten().tolist()
        # print(len(flatten))
    flatten = np.array(flatten)
    # print(flatten.shape)
    return flatten


def l2_norm(gradients: list) -> list[float]:
    norms = []
    for gradient in gradients:
        norms.append(LA.norm(gradient, 2))
    return norms


def total_norm(gradients: list) -> list[float]:
    gradients = np.array(gradients)
    # print("total shape:", np.sum(gradients, axis=0).shape)
    return LA.norm(np.sum(gradients, axis=0), 2)


def pairwise_entropy(
    ndarray1: list,
    ndarray2: list,
) -> list:
    # print(len(ndarray1),len(ndarray2))
    pairwise_entropy_result = [
        [None for j in range(len(ndarray1))] for i in range(len(ndarray2))
    ]
    for i in range(len(ndarray2)):
        for j in range(len(ndarray1)):
            pairwise_entropy_result[i][j] = entropy(ndarray1[j], ndarray2[i])
    return pairwise_entropy_result


class WandbServer(Server):
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        starting_round: int = 0,
        server_rng: ServerRNG,
        strategy: Strategy | None = None,
        history: History | None = None,
        save_parameters_to_file: Callable[
            [Parameters],
            None,
        ],
        save_rng_to_file: Callable[[ServerRNG], None],
        save_history_to_file: Callable[[History], None],
        save_files_per_round: Callable[[int], None],
    ) -> None:
        """Flower server implementation.

        Parameters
        ----------
        client_manager : ClientManager
            Client manager implementation.
        strategy : Optional[Strategy]
            Strategy implementation.
        history : Optional[History]
            History implementation.
        save_parameters_to_file : Callable[[Parameters], None]
            Function to save the parameters to file.
        save_files_per_round : Callable[[int], None]
            Function to save files every round.

        Returns
        -------
        None
        """
        super().__init__(
            client_manager=client_manager,
            strategy=strategy,
        )

        self.history: History | None = history
        self.save_parameters_to_file = save_parameters_to_file
        self.save_files_per_round = save_files_per_round
        self.starting_round = starting_round
        self.server_rng = server_rng
        self.save_rng_to_file = save_rng_to_file
        self.save_history_to_file = save_history_to_file

    # pylint: disable=too-many-locals
    def fit(
        self,
        num_rounds: int,
        timeout: float | None,
    ) -> History:
        """Run federated averaging for a number of rounds.

        Parameters
        ----------
        num_rounds : int
            The number of rounds to run.
        timeout : Optional[float]
            Timeout in seconds.

        Returns
        -------
        History
            The history of the training.
            Potentially using a pre-defined history.
        """
        history = self.history if self.history is not None else History()
        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(
            timeout=timeout,
        )

        if self.starting_round == 0:
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(
                0,
                parameters=self.parameters,
            )
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1]["test_accuracy"],
                )
                history.add_loss_centralized(
                    server_round=0,
                    loss=res[0],
                )
                history.add_metrics_centralized(
                    server_round=0,
                    metrics={
                        "test_accuracy": res[1]["test_accuracy"],
                        "f1_score": res[1]["f1_score"],
                        "confusion_matrix": res[1]["confusion_matrix"],
                        "random_activation": res[1]["random_activation"],
                    },
                )
            # Save initial parameters and files
            self.save_parameters_to_file(self.parameters)
            self.save_rng_to_file(self.server_rng)
            self.save_files_per_round(0)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(self.starting_round + 1, num_rounds + 1):

            combined_metrics = {}
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            original_parameters = deepcopy(self.parameters)
            if res_fit is not None:
                (
                    parameters_prime,
                    fit_metrics,
                    (results, failures),
                ) = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

                print("------------------computing metrics-------------------")
                # print(results[0][1].metrics)
                # print(fit_metrics)
                # print(parameters_to_ndarrays(results[0][1].parameters))
                gradients = [None for _ in range(len(results))]
                for i in range(len(results)):
                    gradients[i] = calculate_pseudo_gradient(
                        parameters_to_ndarrays(results[i][1].parameters),
                        parameters_to_ndarrays(original_parameters),
                    )
                cs_gradients = cosine_similarity(gradients, gradients)
                norm_gradients = l2_norm(gradients)
                total_gradients = total_norm(gradients)
                # print(cs_gradients, norm_gradients, total_gradients)

                original_cen = self.strategy.evaluate(
                    0,
                    parameters=original_parameters,
                )
                # print(original_cen[1].keys())
                # print(results[0][1].metrics.keys())
                # print(results[0][1].metrics["test_activation_mean"])
                activations = [None for _ in range(len(results))]
                activations_mean = [None for _ in range(len(results))]
                for i in range(len(results)):
                    activations[i] = results[i][1].metrics["test_activation"]
                    activations_mean[i] = results[i][1].metrics["test_activation_mean"]
                local_distribution = [None for _ in range(len(results))]
                total_distribution = None
                for i in range(len(results)):
                    local_distribution[i] = results[i][1].metrics["local_distribution"]
                    if total_distribution is None:
                        total_distribution = np.array(
                            results[i][1].metrics["local_distribution"]
                        )
                    else:
                        total_distribution = total_distribution + np.array(
                            results[i][1].metrics["local_distribution"]
                        )
                # print(local_distribution)
                # print(total_distribution)
                # _ = input()
                cs_activation = cosine_similarity(activations, activations)
                cs_activation_mean = cosine_similarity(
                    activations_mean, activations_mean
                )
                cs_server_before = cosine_similarity(
                    activations, [original_cen[1]["test_activation"]]
                )
                cs_server_before_mean = cosine_similarity(
                    activations_mean, [original_cen[1]["test_activation_mean"]]
                )
                kl_activation = pairwise_entropy(activations, activations)
                kl_activation_mean = pairwise_entropy(
                    activations_mean, activations_mean
                )
                kl_server_before = pairwise_entropy(
                    activations, [original_cen[1]["test_activation"]]
                )
                kl_server_before_mean = pairwise_entropy(
                    activations_mean, [original_cen[1]["test_activation_mean"]]
                )
                # print(kl_activation, kl_activation_mean, kl_server_before, kl_server_before_mean)
                combined_metrics["cs_gradients"] = cs_gradients.tolist()
                combined_metrics["norm_gradients"] = norm_gradients
                combined_metrics["total_gradients"] = total_gradients
                combined_metrics["cs_activation"] = cs_activation.tolist()
                combined_metrics["cs_activation_mean"] = cs_activation_mean.tolist()
                combined_metrics["cs_server_before"] = cs_server_before.tolist()
                combined_metrics["cs_server_before_mean"] = (
                    cs_server_before_mean.tolist()
                )
                combined_metrics["local_distribution"] = local_distribution
                combined_metrics["total_distribution"] = total_distribution.tolist()
                combined_metrics["kl_activation"] = kl_activation
                combined_metrics["kl_activation_mean"] = kl_activation_mean
                combined_metrics["kl_server_before"] = kl_server_before
                combined_metrics["kl_server_before_mean"] = kl_server_before_mean

                # _ = input()

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(
                current_round,
                parameters=self.parameters,
            )
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                temp_metrics = {}
                temp_metrics["test_accuracy"] = metrics_cen["test_accuracy"]
                temp_metrics["confusion_matrix"] = metrics_cen["confusion_matrix"]
                temp_metrics["f1_score"] = metrics_cen["f1_score"]
                temp_metrics["random_activation"] = metrics_cen["random_activation"]
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen["test_accuracy"],
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(
                    server_round=current_round,
                    loss=loss_cen,
                )
                history.add_metrics_centralized(
                    server_round=current_round,
                    metrics=temp_metrics,
                )

                cs_server_after = cosine_similarity(
                    activations, [res_cen[1]["test_activation"]]
                )
                cs_server_after_mean = cosine_similarity(
                    activations_mean, [res_cen[1]["test_activation_mean"]]
                )
                actual_gradient = l2_norm([
                    calculate_pseudo_gradient(
                        parameters_to_ndarrays(self.parameters),
                        parameters_to_ndarrays(original_parameters),
                    )
                ])
                kl_server_after = pairwise_entropy(
                    activations, [res_cen[1]["test_activation"]]
                )
                kl_server_after_mean = pairwise_entropy(
                    activations_mean, [res_cen[1]["test_activation_mean"]]
                )
                # print(kl_server_after, kl_server_after_mean)
                # _ = input()
                combined_metrics["cs_server_after"] = cs_server_after.tolist()
                combined_metrics["cs_server_after_mean"] = cs_server_after_mean.tolist()
                combined_metrics["actual_gradient"] = actual_gradient
                combined_metrics["kl_server_after"] = kl_server_after
                combined_metrics["kl_server_after_mean"] = kl_server_after_mean

            # print(combined_metrics)
            # _ = input()
            history.add_metrics_distributed_fit(
                server_round=current_round,
                metrics=combined_metrics,
            )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, (results, failures) = res_fed

                metrics_individual = {}
                for key in results[0][1].metrics.keys():
                    metrics_individual[key] = []
                for result in results:
                    for key, value in result[1].metrics.items():
                        metrics_individual[key].append(value)

                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round,
                        loss=loss_fed,
                    )
                    history.add_metrics_distributed(
                        server_round=current_round,
                        metrics=evaluate_metrics_fed,
                    )

            # Saver round parameters and files
            self.save_parameters_to_file(self.parameters)
            self.save_history_to_file(history)
            self.save_rng_to_file(self.server_rng)
            self.save_files_per_round(current_round)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
