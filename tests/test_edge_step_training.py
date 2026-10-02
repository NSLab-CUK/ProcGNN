from __future__ import annotations

from types import SimpleNamespace

import torch

from process_graph.experiment.edge_step_training import compute_single_edge_loss, train_one_batch_edge_step


class TinyEdgeModel(torch.nn.Module):
    def __init__(self, n_edges: int = 2, dim: int = 3) -> None:
        super().__init__()
        self.pred = torch.nn.Parameter(torch.zeros(n_edges, dim))
        self.forward_count = 0

    def forward(self, batch_data, task_inputs=None):  # noqa: D401
        self.forward_count += 1
        return {"y_edge_pred": self.pred}


class Meta(SimpleNamespace):
    pass


def _cfg(**kw):
    base = dict(
        loss_type_edge_all="smooth_l1",
        shuffle_edges_each_batch=False,
        scheduler_step_unit="epoch",
        edge_weight_default=1.0,
        edge_weight_target_edge=5.0,
        use_all_edge_r2_loss_in_edge_step=False,
        use_target_stream_loss_weighting=True,
        target_stream_loss_weight=5.0,
        target_stream_weighting_mode="unique_target_stream_edge_by_process",
        target_stream_weight_conflict_policy="max",
        allow_yaml_only_target_streams=False,
        target_stream_loss_weights=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _meta():
    return Meta(
        process_id=["Process6", "Process6"],
        canonical_edge_id=["P06_E022", "P06_E999"],
        main_data_stream_key=["PROD", "OTHER"],
        answer_task_names=["", ""],
    )


def _targets(mask_second: bool = True):
    targets = {"edge_stream": torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])}
    if mask_second:
        mask = torch.ones_like(targets["edge_stream"])
    else:
        mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    return targets, {"edge_stream": mask}


def test_single_edge_loss_weights_all_masked_features_not_one_property():
    outputs = {"y_edge_pred": torch.zeros(2, 3)}
    targets, masks = _targets()
    edge_weights = torch.tensor([5.0, 1.0])

    weighted, log = compute_single_edge_loss(
        outputs=outputs,
        targets=targets,
        target_masks=masks,
        edge_id=0,
        train_cfg=_cfg(),
        edge_weight_vector=edge_weights,
        debug=True,
    )
    unweighted, _ = compute_single_edge_loss(
        outputs=outputs,
        targets=targets,
        target_masks=masks,
        edge_id=0,
        train_cfg=_cfg(),
        edge_weight_vector=torch.ones(2),
        debug=True,
    )

    assert weighted is not None
    assert unweighted is not None
    assert torch.allclose(weighted, unweighted * 5.0)
    assert log["mask_sum"] == 3.0
    assert log["pred_e_shape"] == (1, 3)


def test_single_edge_loss_skips_zero_mask_edge():
    outputs = {"y_edge_pred": torch.zeros(2, 3)}
    targets, masks = _targets(mask_second=False)
    loss, log = compute_single_edge_loss(
        outputs=outputs,
        targets=targets,
        target_masks=masks,
        edge_id=1,
        train_cfg=_cfg(),
        edge_weight_vector=torch.ones(2),
        debug=True,
    )
    assert loss is None
    assert log["skipped_edge_reason"] == "mask_sum_zero"


def test_edge_step_reforwards_per_edge_and_steps_only_valid_edges():
    model = TinyEdgeModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    targets, masks = _targets(mask_second=False)

    result = train_one_batch_edge_step(
        model=model,
        batch_data=SimpleNamespace(),
        task_inputs=None,
        targets=targets,
        target_masks=masks,
        optimizer=opt,
        scheduler=None,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        train_cfg=_cfg(),
        device=torch.device("cpu"),
        use_amp=False,
        grad_clip=1.0,
        edge_export_meta=_meta(),
        edge_target_columns=["Temp", "Pres", "Vol_Flow"],
        rng=None,
        debug=True,
    )

    items = result.loss_items
    assert model.forward_count == 2
    assert int(items["edge_update_count"].item()) == 1
    assert int(items["optimizer_step_count"].item()) == 1
    assert int(items["skipped_edge_count"].item()) == 1
    assert int(items["edge_step_target_edge_count"].item()) == 1
    assert items["edge_step_loss_mean"].isfinite()
    assert items["grad_norm"].item() > 0.0
    assert any(row.get("updated_or_skipped") == "skipped" for row in result.debug_rows)


def test_edge_step_repeated_same_batch_loss_remains_finite_and_decreases():
    model = TinyEdgeModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    targets, masks = _targets(mask_second=True)
    losses = []

    for _ in range(8):
        result = train_one_batch_edge_step(
            model=model,
            batch_data=SimpleNamespace(),
            task_inputs=None,
            targets=targets,
            target_masks=masks,
            optimizer=opt,
            scheduler=None,
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            train_cfg=_cfg(),
            device=torch.device("cpu"),
            use_amp=False,
            grad_clip=1.0,
            edge_export_meta=_meta(),
            edge_target_columns=["Temp", "Pres", "Vol_Flow"],
            rng=None,
            debug=True,
        )
        loss = float(result.loss_items["edge_step_loss_mean"].item())
        assert torch.isfinite(torch.tensor(loss))
        assert float(result.loss_items["grad_norm"].item()) > 0.0
        losses.append(loss)

    assert losses[-1] <= losses[0]
