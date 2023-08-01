import tensorflow as tf

from keras_core import backend
from keras_core import ops
from keras_core.optimizers import base_optimizer


class LossScaleOptimizer(base_optimizer.BaseOptimizer):
    def apply(self, grads, trainable_variables=None):
        """
        `grads` should be a list of gradient tensors
        with 1:1 mapping to the list of variables the optimizer was built with.

        `variables` can be provided on the first call to build the optimizer.
        """
        if len(grads) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return

        if trainable_variables is None:
            if not self.built:
                raise ValueError(
                    "When passing `grads` without `variables`, the optimizer "
                    "must already be built on a list of variables. "
                    "Call `optimizer.build(trainable_variables)` first. "
                )
            if len(grads) != len(self._trainable_variables_indices):
                raise ValueError(
                    "When passing `grads` as a list of gradient tensors, the "
                    f"gradients must match `optimizer.variables` one-to-on. "
                    f"Received a list of {len(grads)} gradients, but the "
                    f"optimizer is tracking {len(self._trainable_variables)} "
                    "trainable variables."
                )
            trainable_variables = self._trainable_variables
        else:
            trainable_variables = list(trainable_variables)
            # Optionally build optimizer.
            if not self.built:
                with ops.name_scope(self.name):
                    self.build(trainable_variables)
                self.built = True
            self._check_variables_are_known(trainable_variables)

        with ops.name_scope(self.name):
            # Filter empty gradients.
            grads, trainable_variables = self._filter_empty_gradients(
                grads, trainable_variables
            )
            if len(list(grads)) == 0:
                return

            # Apply clipping and weight decay.
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)

            # Apply gradient updates.
            self._internal_apply_gradients(
                list(zip(grads, trainable_variables))
            )

            # Apply variable constraints after applying gradients.
            for variable in trainable_variables:
                if getattr(variable, "constraint", None) is not None:
                    variable.assign(variable.constraint(variable))
            return self.iterations
