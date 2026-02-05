import tensorflow as tf
import keras as keras


@keras.utils.register_keras_serializable()
class AssignmentLoss(keras.losses.Loss):
    def __init__(self, lambda_excl=0.0, epsilon=1e-7, name="assignment_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_excl = lambda_excl
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        # y_true: (batch, n_jets, 2)
        # y_pred: (batch, n_jets, 2)
        # IMPORTANT: supply mask in y_true[..., 2] or as sample_weight argument

        # Clip probabilities
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0)

        # ---- Cross entropy ----
        ce = -y_true * tf.math.log(y_pred)
        ce = tf.reduce_mean(ce, axis=[1, 2])  # sum over jets and leptons
        ce_loss = ce

        # ---- Exclusivity penalty ----
        # Penalize p(j,1)*p(j,2) for each jet
        if self.lambda_excl > 0:
            p1 = y_pred[:, :, 0]
            p2 = y_pred[:, :, 1]
            overlap = p1 * p2  # shape (batch, jets)
            overlap = overlap
            excl_loss = self.lambda_excl * tf.reduce_sum(
                overlap, axis=-1
            )  # sum over jets
        else:
            excl_loss = 0.0

        return ce_loss + excl_loss

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lambda_excl": self.lambda_excl, "epsilon": self.epsilon})
        return cfg


@keras.utils.register_keras_serializable()
class FocalAssignmentLoss(keras.losses.Loss):
    def __init__(
        self,
        lambda_excl=0.0,
        focal_gamma=0.0,
        epsilon=1e-7,
        name="assignment_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.lambda_excl = lambda_excl
        self.focal_gamma = focal_gamma
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """
        y_true: (batch, n_jets, 2) OR (batch, n_jets, 3) with mask in channel 2
        y_pred: (batch, n_jets, 2)
        """

        # ---------------------------
        # Extract or create mask
        # ---------------------------

        # ---------------------------
        # Numerical safety
        # ---------------------------
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0)

        # ---------------------------
        # Cross entropy
        # ---------------------------
        # base CE
        ce = -y_true * tf.math.log(y_pred)  # (batch, jets, 2)

        # focal weighting only for true classes
        if self.focal_gamma > 0.0:
            p_t = tf.reduce_sum(
                y_true * y_pred, axis=-1, keepdims=True
            )  # (batch, jets, 1)
            focal_factor = tf.pow(1.0 - p_t, self.focal_gamma)
            ce = ce * focal_factor  # (batch, jets, 2)

        # sum over jets + leptons, normalize by #valid jets
        ce_loss = tf.reduce_mean(ce, axis=[1, 2])

        # ---------------------------
        # Exclusivity penalty
        # ---------------------------
        if self.lambda_excl > 0.0:
            p1 = y_pred[:, :, 0]
            p2 = y_pred[:, :, 1]
            overlap = p1 * p2  # (batch, jets)

            # sum jets, mean over batch
            excl = tf.reduce_sum(overlap, axis=1)
            excl_loss = self.lambda_excl * excl
        else:
            excl_loss = 0.0

        # ---------------------------
        # Total loss
        # ---------------------------
        return ce_loss + excl_loss

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "lambda_excl": self.lambda_excl,
                "focal_gamma": self.focal_gamma,
                "epsilon": self.epsilon,
            }
        )
        return cfg


@keras.utils.register_keras_serializable()
class RegressionLoss(keras.losses.Loss):
    def __init__(
        self,
        alpha=1.0,  # floor for denominator (in same units as momenta)
        var_weights=None,  # shape (num_vars,) or None
        epsilon=1e-8,  # numerical safety
        name="regression_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.var_weights = (
            tf.constant(var_weights, dtype=tf.float32)
            if var_weights is not None
            else None
        )

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true, y_pred: shape (batch, n_items, n_vars)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        rel = y_true - y_pred
        sq = tf.square(rel)  # (batch, n_items, n_vars)

        # apply per-variable weights if given
        if self.var_weights is not None:
            # Ensure shape broadcastable: (n_vars,) or (n_vars_of_sq)
            sq = sq * tf.maximum(self.var_weights, self.epsilon)

        # reduce: mean over vars and items, produce per-sample loss
        per_sample = tf.reduce_mean(sq, axis=[1, 2])  # (batch,)

        return per_sample

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "var_weights": (
                    None
                    if self.var_weights is None
                    else self.var_weights.numpy().tolist()
                ),
            }
        )
        return config


@keras.utils.register_keras_serializable()
class PtEtaPhiLoss(keras.losses.Loss):

    def __init__(
        self,
        w_pt=1.0,
        w_eta=1.0,
        w_phi=1.0,
        w_e=0.0,  # weight for energy loss (if using E instead of pz)
        name="PtEtaPhi_loss",
        log_variables=True,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.w_pt = w_pt
        self.w_eta = w_eta
        self.w_phi = w_phi
        self.w_e = w_e
        self.log_variables = log_variables

    def call(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        px_true, py_true, pz_true = y_true[..., 0], y_true[..., 1], y_true[..., 2]
        px_pred, py_pred, pz_pred = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2]

        pt_true = tf.sqrt(tf.square(px_true) + tf.square(py_true) + 1e-8)
        pt_pred = tf.sqrt(tf.square(px_pred) + tf.square(py_pred) + 1e-8)

        eta_true = tf.asinh(pz_true / (pt_true + 1e-8))
        eta_pred = tf.asinh(pz_pred / (pt_pred + 1e-8))

        e_true = tf.sqrt(tf.square(px_true) + tf.square(py_true) + tf.square(pz_true))
        e_pred = tf.sqrt(tf.square(px_pred) + tf.square(py_pred) + tf.square(pz_pred))

        transverse_dot = px_true * px_pred + py_true * py_pred
        pt_product = pt_true * pt_pred + 1e-8
        cos_delta_phi = tf.clip_by_value(transverse_dot / pt_product, -1.0, 1.0)

        if self.log_variables:
            pt_true = tf.math.log(pt_true + 1e-8)
            pt_pred = tf.math.log(pt_pred + 1e-8)
            e_true = tf.math.log(e_true + 1e-8)
            e_pred = tf.math.log(e_pred + 1e-8)
            delta_pt = pt_true - pt_pred
            delta_e = e_true - e_pred
        else:
            delta_pt = pt_true - pt_pred
            delta_e = e_true - e_pred

        delta_eta = eta_true - eta_pred
        delta_phi = tf.acos(cos_delta_phi)
    
        loss_pt = tf.square(delta_pt)
        loss_eta = tf.square(delta_eta)
        loss_phi = tf.square(delta_phi)
        loss_e = tf.square(delta_e)
        total_loss = (
            self.w_pt * loss_pt
            + self.w_eta * loss_eta
            + self.w_phi * loss_phi
            + self.w_e * loss_e
        )
        total_loss = tf.reduce_mean(total_loss, axis=-1)  # mean over items
        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, total_loss.dtype), [-1])
            total_loss = total_loss * sample_weight
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "w_pt": self.w_pt,
                "w_eta": self.w_eta,
                "w_phi": self.w_phi,
                "log_pt": self.log_pt,
            }
        )
        return config


def _get_loss(loss_name):
    if loss_name not in globals():
        raise ValueError(f"Loss '{loss_name}' not found in core.losses.")
    return globals()[loss_name]
