"""Bounded feature sparse GP with an explicit, ROS-independent prediction API.

Uses the feature-augmented RQ/inducing-point structure of the vendored FSGP
reference, not its ROS wrapper. This is a PTR-Nav adaptation, not a claim of
bit-identical paper reproduction. Features and the plane trend use observations
only; query heights are never accepted by the API.
"""

from dataclasses import dataclass
import time

import gpytorch
import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError, cKDTree


def spatial_sample_indices(xyz, budget):
    """Order-independent XY strata; select measurements, never synthesize heights."""
    points = np.asarray(xyz)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    if len(points) <= budget:
        return order
    side = max(2, int(np.sqrt(budget)))
    lower = points[:, :2].min(axis=0)
    span = np.maximum(np.ptp(points[:, :2], axis=0), 1e-6)
    uv = (points[:, :2] - lower) / span * side
    cells = np.minimum(uv.astype(int), side - 1)
    key = cells[:, 1] * side + cells[:, 0]
    distance = np.sum((uv - cells - 0.5) ** 2, axis=1)
    ranked = order[np.lexsort((distance[order], key[order]))]
    _, first = np.unique(key[ranked], return_index=True)
    chosen = ranked[first]
    if len(chosen) < budget:
        remaining = order[~np.isin(order, chosen)]
        extra = remaining[np.linspace(0, len(remaining) - 1, budget - len(chosen), dtype=int)]
        chosen = np.r_[chosen, extra]
    return chosen[np.lexsort((points[chosen, 2], points[chosen, 1], points[chosen, 0]))]


def surrounding_neighbors(tree, query, count, radius, nearest_limit, current_support_distance=None):
    """Keep nearest neighbors unless they miss available enclosing observations.

    Dense one-sided returns can hide the opposite rim of a hole from k-NN.
    Only in that case, reserve neighbors by angular sector within the existing
    physical radius. This selects measurements, never invented support; the GP
    still checks the actual hull, conditioning, and posterior uncertainty.
    """
    distance, indices=tree.query(query,k=count,workers=1)
    delta=tree.data[indices]-query[:,None]
    angles=np.sort(np.arctan2(delta[:,:,1],delta[:,:,0]),axis=1)
    gaps=np.diff(np.c_[angles,angles[:,0]+2*np.pi],axis=1)
    support_distance=distance[:,0] if current_support_distance is None else current_support_distance
    needs=(gaps.max(axis=1)>np.pi+1e-6)&(support_distance<=nearest_limit)&(distance[:,0]>1e-6)
    rows=np.flatnonzero(needs)
    candidates=tree.query_ball_point(query[rows],radius,workers=1)
    for row,available in zip(rows,candidates):
        if len(available)<count:continue
        ids=np.asarray(available,dtype=int)
        offsets=tree.data[ids]-query[row]
        ranges=np.linalg.norm(offsets,axis=1)
        order=np.lexsort((ids,ranges))
        sectors=np.floor((np.arctan2(offsets[:,1],offsets[:,0])+np.pi)*4/np.pi).astype(int)%8
        _,first=np.unique(sectors[order],return_index=True)
        selected=order[np.sort(first)]
        rest=order[~np.isin(order,selected)]
        selected=np.r_[selected,rest[:count-len(selected)]]
        selected=selected[np.lexsort((ids[selected],ranges[selected]))]
        indices[row]=ids[selected];distance[row]=ranges[selected]
    return distance,indices


@dataclass
class FeatureSgpParams:
    device: str = "auto"
    max_input_points: int = 1024
    max_train_points: int = 384
    inducing_points: int = 96
    neighbors: int = 12
    training_steps: int = 12
    learning_rate: float = 0.03
    noise_std_m: float = 0.02
    lengthscale_m: float = 0.7
    support_radius_m: float = 1.2
    accept_std_m: float = 0.10
    query_chunk: int = 512
    unresolved_scale: float = 0.5
    local_conditioning: bool = True
    local_reference_points: int = 16384
    local_neighbors: int = 20
    local_radius_m: float = 1.5
    local_nearest_m: float = .35


@dataclass
class TerrainPrediction:
    mean: np.ndarray
    latent_var: np.ndarray
    observation_var: np.ndarray
    nearest_distance: np.ndarray
    supported: np.ndarray
    accepted: np.ndarray
    device: str
    fit_ms: float
    predict_ms: float
    training_points: int
    gradient_xy: np.ndarray | None = None
    local_conditioned: np.ndarray | None = None
    conditioning_age: np.ndarray | None = None
    conditioning_indices: np.ndarray | None = None
    conditioning_weights: np.ndarray | None = None


class _FeatureKernel(gpytorch.kernels.InducingPointKernel):
    generic_prediction = False

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        if self.generic_prediction:
            from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
            return DefaultPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)
        return super().prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)


class _FeatureGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing, lengthscale):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=4))
        base.base_kernel.lengthscale = torch.tensor([lengthscale, lengthscale, 2.0, 2.0],
                                                    device=train_x.device)
        self.covar_module = _FeatureKernel(base, inducing, likelihood)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class FeatureSgpPredictor:
    def __init__(self, params=None):
        self.params = params or FeatureSgpParams()
        p = self.params
        if (p.device not in ("auto", "cpu", "cuda") or p.max_input_points < 8
                or p.max_train_points < 8 or p.inducing_points < 4 or p.neighbors < 3
                or p.training_steps < 0 or p.query_chunk < 1 or p.noise_std_m <= 0
                or p.support_radius_m <= 0 or p.accept_std_m <= 0 or p.unresolved_scale < 0
                or p.learning_rate <= 0 or p.lengthscale_m <= 0 or p.local_reference_points < 8
                or p.local_neighbors < 10 or p.local_radius_m <= 0 or p.local_nearest_m <= 0
                or not np.isfinite([p.learning_rate, p.noise_std_m, p.lengthscale_m,
                                     p.support_radius_m, p.accept_std_m, p.unresolved_scale,
                                     p.local_radius_m, p.local_nearest_m]).all()):
            raise ValueError("invalid feature SGP parameters")
        if p.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable")
        self.device = torch.device("cuda" if p.device == "auto" and torch.cuda.is_available() else
                                   ("cpu" if p.device == "auto" else p.device))
        self.model = None
        self.fit_ms = 0.0
        self.training_points = 0
        self.local_tree = None
        self.generic_prediction = False
        self.prediction_fallbacks = 0

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _neighbors(self, query, reference, count):
        distances, indices = [], []
        for block in query.split(self.params.query_chunk):
            d = torch.cdist(block, reference)
            values, ids = torch.topk(d, min(count, len(reference)), largest=False, dim=1)
            distances.append(values)
            indices.append(ids)
        return torch.cat(distances), torch.cat(indices)

    def _features(self, xyz):
        _, ids = self._neighbors(xyz[:, :2], xyz[:, :2], self.params.neighbors)
        neighborhood = xyz[ids]
        centered = neighborhood - neighborhood.mean(dim=1, keepdim=True)
        covariance = torch.einsum("bni,bnj->bij", centered, centered) / max(1, ids.shape[1] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        curvature = eigenvalues[:, 0] / eigenvalues.sum(dim=1).clamp_min(1e-8)
        height_change = (neighborhood[:, :, 2] - xyz[:, 2, None]).abs().mean(dim=1)
        return torch.stack((curvature, height_change), dim=1)

    def _encode_query(self, xy):
        distances, ids = self._neighbors(xy, self.reference_xy, min(5, self.params.neighbors))
        weights = 1 / distances.clamp_min(1e-4)
        weights = weights / weights.sum(dim=1, keepdim=True)
        features = (self.reference_features[ids] * weights[:, :, None]).sum(dim=1)
        encoded = torch.cat((xy - self.xy_center,
                             (features - self.feature_center) / self.feature_scale), dim=1)
        return encoded, distances[:, 0]

    def fit(self, observed_xyz):
        xyz = np.asarray(observed_xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("fit accepts observed XYZ only")
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) < 8:
            raise ValueError("at least eight finite observed points are required")
        self._sync()
        started = time.perf_counter()
        self.support_origin = xyz[:, :2].mean(axis=0, dtype=np.float64)
        try:
            self.support_halfspaces = ConvexHull(xyz[:, :2].astype(np.float64) - self.support_origin).equations
        except QhullError as exc:
            raise ValueError("ground observations do not span a two-dimensional surface") from exc
        if self.params.local_conditioning:
            local = xyz[spatial_sample_indices(xyz, self.params.local_reference_points)]
            self.local_tree = cKDTree(local[:, :2])
            self.local_xyz = torch.as_tensor(local, device=self.device)
        else:
            self.local_tree = None
        xyz = xyz[spatial_sample_indices(xyz, self.params.max_input_points)]
        train_ids_np = spatial_sample_indices(xyz, self.params.max_train_points)
        inducing_ids_np = spatial_sample_indices(xyz[train_ids_np], self.params.inducing_points)
        centered_xy = xyz[:, :2].astype(np.float64) - xyz[:, :2].mean(axis=0)
        if np.linalg.eigvalsh(centered_xy.T @ centered_xy)[0] < 1e-6:
            raise ValueError("ground observations do not span a two-dimensional surface")
        with torch.no_grad():
            points = torch.as_tensor(xyz, device=self.device)
            features = self._features(points)
            self.reference_xy = points[:, :2]
            self.reference_features = features
            self.xy_center = self.reference_xy.mean(dim=0)
            self.feature_center = features.mean(dim=0)
            self.feature_scale = features.std(dim=0).clamp_min(0.005)
            # A fitted plane is a data-derived mean, not synthetic robot-ground points.
            design = torch.cat((self.reference_xy - self.xy_center,
                                torch.ones((len(points), 1), device=self.device)), dim=1)
            gram = design.T @ design + torch.eye(3, device=self.device) * 1e-6
            self.plane = torch.linalg.solve(gram, design.T @ points[:, 2])
            residual = points[:, 2] - design @ self.plane
            self.target_scale = residual.std().clamp_min(0.05)
            # Plane uncertainty and unresolved local variation remain even when
            # the inducing approximation reports a tiny conditional variance.
            residual_var = residual.square().sum() / max(1, len(points) - 3)
            self.trend_covariance = torch.linalg.inv(gram) * residual_var.clamp_min(
                self.params.noise_std_m ** 2)
            distances, local_ids = self._neighbors(self.reference_xy, self.reference_xy, 6)
            differences = residual[:, None] - residual[local_ids]
            local_slopes = ((differences.square() - 2 * self.params.noise_std_m ** 2).clamp_min(0)
                            .sqrt() / distances.clamp_min(0.1))
            slopes = local_slopes[distances >= 0.1]
            self.unresolved_slope = torch.quantile(slopes, 0.8) if slopes.numel() else residual_var.new_zeros(())
            train_ids = torch.as_tensor(train_ids_np, device=self.device)
            x = torch.cat((self.reference_xy - self.xy_center,
                           (features - self.feature_center) / self.feature_scale), dim=1)[train_ids]
            y = residual[train_ids] / self.target_scale
            inducing_ids = torch.as_tensor(inducing_ids_np, device=self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-5)).to(self.device)
        likelihood.noise = (self.params.noise_std_m / self.target_scale).square().clamp_min(1e-4)
        self.model = _FeatureGP(x, y, likelihood, x[inducing_ids].clone(),
                               self.params.lengthscale_m).to(self.device)
        self.model.covar_module.generic_prediction = self.generic_prediction
        self.model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.model)
        with gpytorch.settings.cholesky_jitter(1e-4):
            for _ in range(self.params.training_steps):
                optimizer.zero_grad()
                loss = -objective(self.model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite SGP training objective")
                loss.backward()
                optimizer.step()
        self.model.eval()
        likelihood.eval()
        self.training_points = len(x)
        self._sync()
        self.fit_ms = (time.perf_counter() - started) * 1000
        return self

    def _local_surface(self, xy, global_mean, global_var, nearest, *, latent_conditioning=False,
                       current_support_distance=None):
        if self.local_tree is None or len(self.local_xyz) < 10:
            return global_mean, global_var, nearest, torch.zeros(len(xy), dtype=torch.bool, device=self.device)
        query=xy.detach().cpu().numpy()
        count=min(self.params.local_neighbors,len(self.local_xyz))
        if latent_conditioning:
            distance,ids=surrounding_neighbors(self.local_tree,query,count,
                self.params.local_radius_m,self.params.local_nearest_m,current_support_distance)
            self.local_conditioning_indices=ids
        else:
            distance,ids=self.local_tree.query(query,k=count,workers=1)
        support_distance=distance[:,0] if current_support_distance is None else current_support_distance
        local_mean, local_var, applied, source_weights = [], [], [], []
        noise = self.params.noise_std_m ** 2
        # This is a local measured-surface GP, selected instead of the global
        # posterior, not precision fusion of two models trained on the same data.
        kernel = gpytorch.kernels.RBFKernel().to(self.device)
        kernel.lengthscale = .7
        for start in range(0, len(xy), 4096):
            stop = min(len(xy), start + 4096)
            index = torch.as_tensor(ids[start:stop], device=self.device)
            points = self.local_xyz[index]
            delta = points[:, :, :2] - xy[start:stop, None]
            scale = delta.square().sum(dim=2).max(dim=1).values.sqrt().clamp_min(.1)
            uv = delta / scale[:, None, None]
            x, y = uv[:, :, 0], uv[:, :, 1]
            design = torch.stack((torch.ones_like(x), x, y, x*x, x*y, y*y), dim=2)
            weights = 1 / (uv.square().sum(dim=2) + .2)
            gram = design.transpose(1, 2) @ (design * weights[:, :, None])
            regularizer = torch.diag(gram.new_tensor([1e-6, 1e-5, 1e-5, .002, .002, .002]))
            gram = gram + regularizer
            rhs = design.transpose(1, 2) @ (points[:, :, 2:3] * weights[:, :, None])
            coefficients = torch.linalg.solve(gram, rhs)
            residual = points[:, :, 2] - (design @ coefficients).squeeze(2)
            amplitude = (residual.square().sum(dim=1) / max(1, len(index[0]) - 6)).clamp_min(noise)
            covariance = kernel(uv, uv).to_dense() * amplitude[:, None, None]
            covariance = covariance + torch.eye(uv.shape[1], device=self.device)[None] * noise
            cross = kernel(uv, torch.zeros_like(uv[:, :1])).to_dense().squeeze(2) * amplitude[:, None]
            solves = torch.linalg.solve(covariance, torch.stack((residual, cross), dim=2))
            mean = coefficients[:, 0, 0] + (cross * solves[:, :, 0]).sum(dim=1)
            gram_inverse = torch.linalg.inv(gram)
            trend_var = gram_inverse[:, 0, 0].clamp_min(0) * amplitude
            variance = (amplitude - (cross * solves[:, :, 1]).sum(dim=1) + trend_var).clamp_min(noise)
            leverage = (design * (design @ gram_inverse)).sum(dim=2) * weights
            loo_residual = residual / (1-leverage).clamp_min(.15)
            # Leave-one-observation-out lack of fit is measurement-only evidence.
            # Do not shrink intervals just because a dense local fit interpolates.
            lack_of_fit = (loo_residual.square().mean(dim=1)-noise).clamp_min(0)
            variance = variance + lack_of_fit
            if latent_conditioning:
                # Universal kriging: uncertain polynomial trend + residual GP.
                # Latent ground variance is NOT observation noise. Keeping the
                # latter as a hard latent floor loses the benefit of conditioning.
                inverse = torch.linalg.inv(covariance)
                inverse_design = inverse @ design
                gls_gram = design.transpose(1, 2) @ inverse_design
                eigen = torch.linalg.eigvalsh(gls_gram)
                identifiable = eigen[:, 0] > eigen[:, -1]*1e-7
                safe_gram = torch.where(identifiable[:,None,None],gls_gram,torch.eye(6,device=self.device)[None])
                gls_inverse = torch.linalg.inv(safe_gram)
                center_z = points[:, :, 2].mean(dim=1)
                centered_z = points[:, :, 2]-center_z[:,None]
                gls_coeff = gls_inverse @ (inverse_design.transpose(1, 2) @ centered_z[:, :, None])
                gls_residual = centered_z - (design @ gls_coeff).squeeze(2)
                inverse_cross = (inverse @ cross[:, :, None]).squeeze(2)
                mean = center_z + gls_coeff[:, 0, 0] + (inverse_cross*gls_residual).sum(dim=1)
                contrast = -(design.transpose(1, 2) @ inverse_cross[:, :, None]).squeeze(2)
                contrast[:, 0] += 1
                kriging_weights=inverse_cross+(inverse_design@gls_inverse@contrast[:,:,None]).squeeze(2)
                source_weights.append(torch.where(identifiable[:,None],kriging_weights,0.).cpu().numpy())
                trend_uncertainty = torch.einsum("bi,bij,bj->b", contrast, gls_inverse, contrast)
                variance = (amplitude-(cross*inverse_cross).sum(dim=1)+trend_uncertainty).clamp_min(1e-10)
                precision = inverse - inverse_design @ gls_inverse @ inverse_design.transpose(1, 2)
                diagonal = torch.diagonal(precision, dim1=1, dim2=2).clamp_min(1e-6)
                loo = (precision @ centered_z[:, :, None]).squeeze(2)/diagonal
                variance += (loo.square()-1/diagonal).mean(dim=1).clamp_min(0)
                # Degenerate polynomial trends cannot get a confident posterior.
                variance = torch.where(identifiable, variance, torch.inf)
            radius = torch.as_tensor(distance[start:stop, -1], device=self.device)
            # Axis-wise bracketing also admits diagonal extrapolation. A query
            # lies in a 2D neighborhood's convex hull iff no angular gap exceeds pi.
            angles = torch.sort(torch.atan2(y, x), dim=1).values
            gaps = torch.cat((angles[:, 1:] - angles[:, :-1],
                              angles[:, :1] + 2 * torch.pi - angles[:, -1:]), dim=1)
            surrounded = (gaps.max(dim=1).values <= torch.pi + 1e-6) | \
                (delta.square().sum(dim=2).min(dim=1).values <= 1e-12)
            geometry = uv.transpose(1, 2) @ uv
            eigenvalues = torch.linalg.eigvalsh(geometry)
            usable = surrounded & (radius <= self.params.local_radius_m) & \
                (torch.as_tensor(support_distance[start:stop], device=self.device) <= self.params.local_nearest_m) & \
                (eigenvalues[:, 0] >= .01 * eigenvalues[:, 1]) & torch.isfinite(mean) & \
                (variance <= self.params.accept_std_m ** 2)
            local_mean.append(torch.where(usable, mean, global_mean[start:stop]))
            local_var.append(torch.where(usable, variance, global_var[start:stop]))
            applied.append(usable)
        raw_nearest = torch.as_tensor(support_distance, dtype=nearest.dtype, device=self.device)
        if latent_conditioning:self.local_conditioning_weights=np.concatenate(source_weights,axis=0)
        return torch.cat(local_mean), torch.cat(local_var), raw_nearest, torch.cat(applied)

    def _posterior(self, xy):
        encoded, nearest = self._encode_query(xy)
        means, latent_vars = [], []
        for block in encoded.split(self.params.query_chunk):
            try:
                with gpytorch.settings.fast_pred_var(not self.generic_prediction):
                    latent = self.model(block)
            except ValueError as exc:
                if self.generic_prediction or "Expected SGPR output to be" not in str(exc):
                    raise
                # The optimized SGPR strategy rejects some equivalent lazy
                # operator sums. Retry the same learned kernel and likelihood
                # with the library's generic exact conditioner, on this device.
                self.generic_prediction = True
                self.prediction_fallbacks += 1
                self.model.covar_module.generic_prediction = True
                self.model.prediction_strategy = None
                with gpytorch.settings.fast_pred_var(False):
                    latent = self.model(block)
            means.append(latent.mean)
            if self.generic_prediction:
                kernel=self.model.covar_module
                cross=kernel(block,self.model.train_inputs[0]).to_dense()
                solved=self.model.prediction_strategy.lik_train_train_covar.solve(cross.mT)
                # SGPR uses the full base-kernel prior on test points. Generic
                # joint slicing can lose its inducing diagonal correction.
                prior=kernel.base_kernel(block,block,diag=True)
                latent_vars.append((prior-(cross*solved.mT).sum(dim=1)).clamp_min(1e-10))
            else:
                latent_vars.append(latent.variance)
        trend = (xy - self.xy_center) @ self.plane[:2] + self.plane[2]
        design = torch.cat((xy - self.xy_center, torch.ones_like(nearest[:, None])), dim=1)
        trend_var = torch.einsum("ni,ij,nj->n", design, self.trend_covariance, design).clamp_min(0)
        extra_var = trend_var + (nearest * self.unresolved_slope * self.params.unresolved_scale).square()
        latent_var = (torch.cat(latent_vars).clamp_min(1e-10) * self.target_scale.square()
                      + extra_var).clamp_min(self.params.noise_std_m ** 2)
        observation_noise = self.model.likelihood.noise * self.target_scale.square()
        return (torch.cat(means) * self.target_scale + trend,
                latent_var, latent_var + observation_noise, nearest)

    def condition_local(self, observed_xyz, query_xy, *, current_support_distance=None):
        """Recondition the local GP on this observation set, without a stale fit.

        Use a dedicated predictor instance, not the asynchronous global model.
        Unsupported queries stay NaN; query heights are not an input.
        Optional distances come from corroborating current raw returns, not
        snapped grid centers. They do not replace hull or uncertainty checks.
        """
        xyz = np.asarray(observed_xyz, dtype=np.float32)
        query = np.asarray(query_xy, dtype=np.float32)
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 10
                or query.ndim != 2 or query.shape[1] != 2 or not len(query)
                or not np.isfinite(xyz).all() or not np.isfinite(query).all()):
            raise ValueError("local conditioning requires finite observed XYZ and query XY")
        if current_support_distance is not None:
            current_support_distance=np.asarray(current_support_distance,dtype=np.float64)
            if (current_support_distance.shape!=(len(query),) or
                    not np.isfinite(current_support_distance).all() or (current_support_distance<0).any()):
                raise ValueError("current support distances must be finite, nonnegative, and query-aligned")
        self._sync()
        started = time.perf_counter()
        local = xyz[spatial_sample_indices(xyz, self.params.local_reference_points)]
        self.local_tree = cKDTree(local[:, :2])
        self.local_xyz = torch.as_tensor(local, device=self.device)
        self.local_conditioning_indices=None
        self.local_conditioning_weights=None
        with torch.no_grad():
            xy = torch.as_tensor(query, device=self.device)
            mean, variance, nearest, accepted = self._local_surface(
                xy, torch.full((len(xy),), torch.nan, device=self.device),
                torch.full((len(xy),), torch.inf, device=self.device),
                torch.full((len(xy),), torch.inf, device=self.device), latent_conditioning=True,
                current_support_distance=current_support_distance)
            arrays = [a.cpu().numpy() for a in
                (mean, variance, variance + self.params.noise_std_m ** 2,
                 nearest, accepted, accepted)]
        self._sync()
        elapsed = (time.perf_counter() - started) * 1000
        return TerrainPrediction(*arrays, str(self.device), 0., elapsed, len(local),
                                 local_conditioned=accepted.cpu().numpy(),
                                 conditioning_indices=self.local_conditioning_indices,
                                 conditioning_weights=self.local_conditioning_weights)

    def predict(self, query_xy, *, gradients=False, derivative_step_m=0.05):
        if self.model is None:
            raise RuntimeError("fit must precede predict")
        physical_query = np.asarray(query_xy, dtype=np.float64)
        query = physical_query.astype(np.float32)
        if (query.ndim != 2 or query.shape[1] != 2 or not len(query)
                or not np.isfinite(query).all() or derivative_step_m <= 0):
            raise ValueError("query must be a nonempty finite XY array")
        self._sync()
        started = time.perf_counter()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
            xy = torch.as_tensor(query, device=self.device)
            mean, latent_var, observation_var, nearest = self._posterior(xy)
            mean, latent_var, nearest, local = self._local_surface(xy, mean, latent_var, nearest)
            observation_var = latent_var + self.model.likelihood.noise * self.target_scale.square()
            equations = self.support_halfspaces
            # GPU rounding must not move a physical query across the support boundary.
            inside_hull = np.all((physical_query - self.support_origin) @ equations[:, :2].T
                                 + equations[:, 2] <= 1e-6, axis=1)
            supported = (nearest <= self.params.support_radius_m) & torch.as_tensor(inside_hull, device=self.device)
            accepted = supported & (latent_var.sqrt() <= self.params.accept_std_m)
            gradient = None
            if gradients:
                components = []
                for axis in range(2):
                    delta = torch.zeros_like(xy)
                    delta[:, axis] = derivative_step_m
                    # Recompute query features too: this is a spatial derivative,
                    # not a derivative at frozen curvature/gradient coordinates.
                    plus_posterior = self._posterior(xy + delta)
                    minus_posterior = self._posterior(xy - delta)
                    plus = self._local_surface(xy + delta, plus_posterior[0], plus_posterior[1], plus_posterior[3])[0]
                    minus = self._local_surface(xy - delta, minus_posterior[0], minus_posterior[1], minus_posterior[3])[0]
                    components.append((plus - minus) / (2 * derivative_step_m))
                gradient = torch.stack(components, dim=1).cpu().numpy()
            arrays = [a.cpu().numpy() for a in (mean, latent_var, observation_var, nearest,
                                               supported, accepted)]
        self._sync()
        predict_ms = (time.perf_counter() - started) * 1000
        return TerrainPrediction(*arrays, str(self.device), self.fit_ms, predict_ms,
                                 self.training_points, gradient, local.cpu().numpy())
