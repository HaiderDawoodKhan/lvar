import unittest

import torch

from lvar.grpo_training import (
    clipped_grpo_loss,
    normalize_group_rewards,
    select_controller_action,
    set_phase5_trainable,
    rollout_phase5,
    target_logprob,
)
from lvar.utils import ACTION_PATCH, ACTION_STOP
from lvar_scripts.train_grpo import (
    asymmetric_baseline_weight,
    build_constant_with_warmup_scheduler,
    compute_correctness_only_reward,
    compute_grpo_policy_loss,
    iter_minibatches,
)
from test_model import build_model


class GRPOTrainingTests(unittest.TestCase):
    def test_patch_rollout_produces_differentiable_policy_loss(self):
        model = build_model(controller_context_window=1, max_steps=1, region_window=2)
        model.train()

        with torch.no_grad():
            for parameter in model.controller.parameters():
                parameter.zero_()
            model.controller.type_head.bias.fill_(-10.0)
            model.controller.type_head.bias[ACTION_PATCH] = 10.0

        def fake_decode(state, labels=None):
            del labels
            return {
                "answer": "yes",
                "generated_text": "<answer>yes</answer>",
                "generated_ids": [1],
                "decode_prefix_length": state["inputs_embeds"].size(1),
                "final_sequence_length": state["inputs_embeds"].size(1),
            }

        model.decode_answer = fake_decode
        rollout = model.forward("image", "question", sample_actions=True)

        self.assertEqual(rollout["trace"][0]["action_id"], ACTION_PATCH)
        self.assertEqual(
            rollout["trace"][0]["sequence_length_after"] - rollout["trace"][0]["sequence_length_before"],
            1,
        )
        self.assertTrue(rollout["action_log_prob_sum"].requires_grad)

        loss = compute_grpo_policy_loss(torch.tensor([1.0]), [rollout])
        self.assertIsNotNone(loss)
        loss.backward()

        self.assertIsNotNone(model.controller.type_head.bias.grad)
        self.assertTrue(torch.isfinite(model.controller.type_head.bias.grad).all())

    def test_grpo_policy_loss_returns_none_without_action_log_probs(self):
        loss = compute_grpo_policy_loss(torch.tensor([1.0]), [{"action_log_prob_sum": None}])

        self.assertIsNone(loss)

    def test_clipped_grpo_loss_uses_old_and_current_log_probs(self):
        rollout = {"old_log_probs": [torch.log(torch.tensor(0.5))]}
        current = [[torch.log(torch.tensor(0.7))]]

        loss = clipped_grpo_loss(torch.tensor([1.0]), [rollout], current, clip_epsilon=0.2)

        self.assertTrue(torch.allclose(loss, torch.tensor(-1.2)))

    def test_group_advantage_is_zero_when_rewards_identical(self):
        rewards = torch.tensor([0.5, 0.5, 0.5])

        advantages = normalize_group_rewards(rewards, epsilon=1e-6)

        self.assertTrue(torch.equal(advantages, torch.zeros_like(rewards)))

    def test_asymmetric_weighting_applies_after_normalization(self):
        rewards = torch.tensor([0.0, 1.0])
        advantages = normalize_group_rewards(rewards, epsilon=1e-6)
        weights = torch.tensor(
            [
                asymmetric_baseline_weight(0.0, 0.0, improve_weight=2.0, miss_weight=1.0),
                asymmetric_baseline_weight(0.0, 1.0, improve_weight=2.0, miss_weight=1.0),
            ]
        )

        weighted = advantages * weights

        self.assertTrue(torch.allclose(weighted, advantages * torch.tensor([1.0, 2.0])))

    def test_correctness_only_reward_discards_shaping_terms(self):
        rollout = {
            "actions": [{"type": "THINK"}, {"type": "PATCH", "patch_idx": 1}],
            "selected_visual_actions": [{"type": "PATCH", "patch_idx": 1}],
            "num_steps": 2,
            "stopped": False,
        }

        reward = compute_correctness_only_reward(1.0, rollout)

        self.assertEqual(reward["reward"], 1.0)
        self.assertEqual(reward["r_correct"], 1.0)
        self.assertEqual(reward["r_logp"], 0.0)
        self.assertEqual(reward["r_cf"], 0.0)
        self.assertEqual(reward["r_stop"], 0.0)
        self.assertEqual(reward["r_visual"], 0.0)
        self.assertEqual(reward["r_early_stop"], 0.0)
        self.assertEqual(reward["r_think_once"], 0.0)
        self.assertEqual(reward["think_count"], 1.0)
        self.assertEqual(reward["visual_count"], 1.0)

    def test_constant_scheduler_uses_linear_warmup_then_stays_constant(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=3e-5)
        scheduler = build_constant_with_warmup_scheduler(optimizer, total_steps=4, warmup_ratio=0.5)

        initial_lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        warmup_end_lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        constant_lr = scheduler.get_last_lr()[0]

        self.assertAlmostEqual(initial_lr, 1.5e-5)
        self.assertAlmostEqual(warmup_end_lr, 3e-5)
        self.assertAlmostEqual(constant_lr, 3e-5)

    def test_iter_minibatches_splits_rollouts_by_requested_size(self):
        minibatches = list(iter_minibatches(list(range(10)), 4))

        self.assertEqual(minibatches, [(0, [0, 1, 2, 3]), (4, [4, 5, 6, 7]), (8, [8, 9])])

    def test_patch_sampling_masks_already_selected_patches(self):
        model = build_model(controller_context_window=1, max_steps=1)
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        bank = model.build_visual_bank(projected)
        state = model.build_initial_state(prepared)

        def controller_forward(state_hidden, step_hidden, patch_bank, selected_mask, act_hidden=None):
            del state_hidden, step_hidden, selected_mask, act_hidden
            type_logits = torch.full((1, 3), -10.0)
            type_logits[0, ACTION_PATCH] = 10.0
            patch_logits = torch.zeros(1, patch_bank.size(1))
            patch_logits[0, 3] = 10.0
            patch_logits[0, 2] = 9.0
            return type_logits, patch_logits

        model.controller.forward = controller_forward

        action, _, _ = select_controller_action(
            model,
            state,
            bank,
            step_idx=0,
            temperature=1.0,
            selected_patches={3},
            sample=False,
        )

        self.assertEqual(action["type"], "PATCH")
        self.assertEqual(action["patch_idx"], 2)

    def test_phase5_rollout_starts_from_full_context(self):
        model = build_model(controller_context_window=1, max_steps=1)
        called = {"full": 0, "coarse": 0}
        original_full = model.build_initial_state
        original_coarse = model.build_coarse_initial_state

        def build_full(batch):
            called["full"] += 1
            return original_full(batch)

        def build_coarse(batch, bank):
            called["coarse"] += 1
            return original_coarse(batch, bank)

        def controller_forward(state_hidden, step_hidden, patch_bank, selected_mask, act_hidden=None):
            del state_hidden, step_hidden, patch_bank, selected_mask, act_hidden
            type_logits = torch.full((1, 3), -10.0)
            type_logits[0, ACTION_STOP] = 10.0
            return type_logits, torch.zeros(1, 4)

        model.build_initial_state = build_full
        model.build_coarse_initial_state = build_coarse
        model.controller.forward = controller_forward
        model.decode_answer = lambda state, labels=None: {
            "answer": "yes",
            "generated_text": "<answer>yes</answer>",
            "generated_ids": [1],
            "decode_prefix_length": state["inputs_embeds"].size(1),
            "final_sequence_length": state["inputs_embeds"].size(1),
        }

        rollout_phase5(model, "image", "question", max_controller_steps=1, temperature=1.0)

        self.assertEqual(called["full"], 1)
        self.assertEqual(called["coarse"], 0)

    def test_no_stop_rollout_is_marked_for_penalty_reward(self):
        model = build_model(controller_context_window=1, max_steps=1)

        def controller_forward(state_hidden, step_hidden, patch_bank, selected_mask, act_hidden=None):
            del state_hidden, step_hidden, patch_bank, selected_mask, act_hidden
            type_logits = torch.full((1, 3), -10.0)
            type_logits[0, ACTION_PATCH] = 10.0
            return type_logits, torch.zeros(1, 4)

        model.controller.forward = controller_forward
        model.decode_answer = lambda state, labels=None: {
            "answer": "yes",
            "generated_text": "<answer>yes</answer>",
            "generated_ids": [1],
            "decode_prefix_length": state["inputs_embeds"].size(1),
            "final_sequence_length": state["inputs_embeds"].size(1),
        }

        rollout = rollout_phase5(model, "image", "question", max_controller_steps=1, temperature=1.0)

        self.assertFalse(rollout["stopped"])
        self.assertEqual(rollout["actions"][0]["type"], "PATCH")

    def test_gold_answer_logprob_is_length_normalized_and_finite(self):
        model = build_model(controller_context_window=1)
        state, _ = model.build_initial_state(model.prepare_inputs("image", "question")), None

        logp = target_logprob(model, state, "A")

        self.assertTrue(torch.isfinite(logp))

    def test_phase5_trainable_filter_freezes_lora_and_trains_controller_only(self):
        model = build_model(controller_context_window=1)
        model.backbone.lora_adapter = torch.nn.Parameter(torch.ones(()))

        set_phase5_trainable(model)
        trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        self.assertNotIn("backbone.lora_adapter", trainable_names)
        self.assertTrue(any(name.startswith("controller.") for name in trainable_names))
        self.assertIn("step_embedding.weight", trainable_names)
        self.assertIn("controller_state_norm.weight", trainable_names)
        self.assertNotIn("latent_token", trainable_names)


if __name__ == "__main__":
    unittest.main()
