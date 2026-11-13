import argparse
from collections import Counter, OrderedDict, defaultdict

import numpy as np


def normalize_org_id(value):
    return str(value).strip()


def _require_predict_helpers():
    from predict import load_data_and_model_flexible, load_mapping_dicts, normalize_org_id as _normalize_org_id
    return load_data_and_model_flexible, load_mapping_dicts, _normalize_org_id


DEFAULT_SPLIT_RATIO = 0.2
DEFAULT_K = 10
DEFAULT_BATCH_SIZE = 256


def calculate_ndcg(recommended_items, ground_truth_items, k):
    true_set = set(ground_truth_items)
    if not true_set:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in true_set:
            dcg += 1.0 / np.log2(i + 2)

    idcg = 0.0
    for i in range(min(len(true_set), k)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def _load_user_item_file(file_path):
    user_items = defaultdict(list)
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            user_id = int(tokens[0])
            user_items[user_id].extend(int(x) for x in tokens[1:])
    return user_items


def _load_internal_rec_file(file_path):
    return _load_user_item_file(file_path)


def _extract_user_items(inter_feat, uid_field, iid_field):
    user_items = defaultdict(list)
    for user_id, item_id in zip(inter_feat[uid_field].tolist(), inter_feat[iid_field].tolist()):
        if user_id > 0 and item_id > 0:
            user_items[int(user_id)].append(int(item_id))
    return user_items


def _build_org_to_internal(dataset, field_name):
    import torch

    total = dataset.user_num if field_name == dataset.uid_field else dataset.item_num
    internal_ids = torch.arange(total)
    org_tokens = dataset.id2token(field_name, internal_ids)
    mapping = {}
    for internal_id, org_token in enumerate(org_tokens):
        if org_token == '[PAD]':
            continue
        mapping[normalize_org_id(org_token)] = internal_id
    return mapping


def _load_recbole_rec_file(rec_file, user_map_file, item_map_file, dataset, uid_field, iid_field):
    _, load_mapping_dicts, normalize_fn = _require_predict_helpers()
    _, user_remap_to_org = load_mapping_dicts(user_map_file)
    _, item_remap_to_org = load_mapping_dicts(item_map_file)
    global normalize_org_id
    normalize_org_id = normalize_fn
    user_org_to_internal = _build_org_to_internal(dataset, uid_field)
    item_org_to_internal = _build_org_to_internal(dataset, iid_field)

    rec_user_items = {}
    with open(rec_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue

            user_remap_id = int(tokens[0])
            user_org_id = user_remap_to_org.get(user_remap_id)
            if user_org_id is None:
                continue

            internal_user_id = user_org_to_internal.get(user_org_id)
            if internal_user_id is None:
                continue

            internal_item_ids = []
            for item_token in tokens[1:]:
                item_org_id = item_remap_to_org.get(int(item_token))
                if item_org_id is None:
                    continue
                internal_item_id = item_org_to_internal.get(item_org_id)
                if internal_item_id is not None:
                    internal_item_ids.append(internal_item_id)

            rec_user_items[int(internal_user_id)] = internal_item_ids
    return rec_user_items


def _select_eval_split(valid_data, test_data, uid_field):
    test_inter = test_data.dataset.inter_feat
    if len(test_inter[uid_field]) > 0:
        return test_data, 'test'
    return valid_data, 'valid'


def _group_users_and_items(train_user_items, split_ratio):
    user_activity = Counter()
    item_popularity = Counter()

    for user_id, items in train_user_items.items():
        user_activity[user_id] = len(items)
        item_popularity.update(items)

    if not user_activity or not item_popularity:
        return set(), set(), set(), set()

    activity_threshold = np.quantile(
        np.array(list(user_activity.values()), dtype=np.float32),
        1.0 - split_ratio,
    )
    popularity_threshold = np.quantile(
        np.array(list(item_popularity.values()), dtype=np.float32),
        1.0 - split_ratio,
    )

    active_users = {u for u, cnt in user_activity.items() if cnt >= activity_threshold}
    inactive_users = set(user_activity.keys()) - active_users
    popular_items = {i for i, cnt in item_popularity.items() if cnt >= popularity_threshold}
    unpopular_items = set(item_popularity.keys()) - popular_items
    return active_users, inactive_users, popular_items, unpopular_items


def _build_grouped_ground_truth(test_user_items, popular_items, unpopular_items):
    ground_truth_by_group = {}
    for user_id, items in test_user_items.items():
        grouped = {"popular": set(), "unpopular": set()}
        for item_id in items:
            if item_id in popular_items:
                grouped["popular"].add(item_id)
            elif item_id in unpopular_items:
                grouped["unpopular"].add(item_id)
        ground_truth_by_group[user_id] = grouped
    return ground_truth_by_group


def evaluate_fairness_from_topk(train_user_items, test_user_items, rec_user_items, split_ratio=DEFAULT_SPLIT_RATIO, k=DEFAULT_K):
    active_users, inactive_users, popular_items, unpopular_items = _group_users_and_items(
        train_user_items, split_ratio
    )
    ground_truth_by_group = _build_grouped_ground_truth(test_user_items, popular_items, unpopular_items)

    user_metrics = {
        "active": {"ndcg_pop": [], "hit_pop": [], "ndcg_unpop": [], "hit_unpop": [], "count": 0},
        "inactive": {"ndcg_pop": [], "hit_pop": [], "ndcg_unpop": [], "hit_unpop": [], "count": 0},
    }
    item_exposure_counts = Counter()
    overall_ndcgs = []
    overall_hits = []

    for user_id, rec_items in rec_user_items.items():
        for item_id in rec_items[:k]:
            item_exposure_counts[item_id] += 1

        user_gt = ground_truth_by_group.get(user_id)
        if not user_gt:
            continue

        true_popular = user_gt["popular"]
        true_unpopular = user_gt["unpopular"]
        true_all = true_popular | true_unpopular
        if not true_all:
            continue

        rec_k = rec_items[:k]
        rec_k_set = set(rec_k)
        overall_ndcgs.append(calculate_ndcg(rec_items, true_all, k))
        overall_hits.append(len(rec_k_set & true_all))

        if user_id in active_users:
            metrics = user_metrics["active"]
        elif user_id in inactive_users:
            metrics = user_metrics["inactive"]
        else:
            continue

        metrics["count"] += 1
        if true_popular:
            metrics["ndcg_pop"].append(calculate_ndcg(rec_items, true_popular, k))
            metrics["hit_pop"].append(len(rec_k_set & true_popular))
        if true_unpopular:
            metrics["ndcg_unpop"].append(calculate_ndcg(rec_items, true_unpopular, k))
            metrics["hit_unpop"].append(len(rec_k_set & true_unpopular))

    def avg(values):
        return float(np.mean(values)) if values else 0.0

    active_hit_pop = avg(user_metrics["active"]["hit_pop"])
    active_hit_unpop = avg(user_metrics["active"]["hit_unpop"])
    inactive_hit_pop = avg(user_metrics["inactive"]["hit_pop"])
    inactive_hit_unpop = avg(user_metrics["inactive"]["hit_unpop"])

    active_ndcg_pop = avg(user_metrics["active"]["ndcg_pop"])
    active_ndcg_unpop = avg(user_metrics["active"]["ndcg_unpop"])
    inactive_ndcg_pop = avg(user_metrics["inactive"]["ndcg_pop"])
    inactive_ndcg_unpop = avg(user_metrics["inactive"]["ndcg_unpop"])

    ndcg_groups = [active_ndcg_pop, active_ndcg_unpop, inactive_ndcg_pop, inactive_ndcg_unpop]
    hit_groups = [active_hit_pop, active_hit_unpop, inactive_hit_pop, inactive_hit_unpop]
    ndcg_max, ndcg_min = max(ndcg_groups), min(ndcg_groups)
    hit_max, hit_min = max(hit_groups), min(hit_groups)

    pop_exposure = sum(item_exposure_counts[i] for i in popular_items)
    unpop_exposure = sum(item_exposure_counts[i] for i in unpopular_items)

    return OrderedDict([
        ("g_active_users", len(active_users)),
        ("g_inactive_users", len(inactive_users)),
        ("g_popular_items", len(popular_items)),
        ("g_unpopular_items", len(unpopular_items)),
        (f"active_pop_ndcg@{k}", active_ndcg_pop),
        (f"active_pop_hit@{k}", active_hit_pop),
        (f"active_unpop_ndcg@{k}", active_ndcg_unpop),
        (f"active_unpop_hit@{k}", active_hit_unpop),
        (f"inactive_pop_ndcg@{k}", inactive_ndcg_pop),
        (f"inactive_pop_hit@{k}", inactive_hit_pop),
        (f"inactive_unpop_ndcg@{k}", inactive_ndcg_unpop),
        (f"inactive_unpop_hit@{k}", inactive_hit_unpop),
        ("popular_total_exposure", int(pop_exposure)),
        ("unpopular_total_exposure", int(unpop_exposure)),
        ("popular_avg_exposure", pop_exposure / len(popular_items) if popular_items else 0.0),
        ("unpopular_avg_exposure", unpop_exposure / len(unpopular_items) if unpopular_items else 0.0),
        (f"overall_ndcg@{k}", avg(overall_ndcgs)),
        (f"overall_hit@{k}", avg(overall_hits)),
        (f"ndcg_mmd@{k}", ndcg_max - ndcg_min),
        (f"ndcg_mmr@{k}", (ndcg_max / ndcg_min) if ndcg_min > 0 else float("inf")),
        (f"ndcg_lm@{k}", ndcg_min),
        (f"hit_mmd@{k}", hit_max - hit_min),
        (f"hit_mmr@{k}", (hit_max / hit_min) if hit_min > 0 else float("inf")),
        (f"hit_lm@{k}", hit_min),
    ])


def evaluate_fairness_with_model(model, config, train_data, eval_data, k=DEFAULT_K, split_ratio=DEFAULT_SPLIT_RATIO, batch_size=None):
    import torch
    from recbole.data.interaction import Interaction

    dataset = train_data.dataset
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    if batch_size is None:
        score_budget = int(config["eval_batch_size"]) if "eval_batch_size" in config else DEFAULT_BATCH_SIZE
    else:
        score_budget = None

    train_user_items = _extract_user_items(train_data.dataset.inter_feat, uid_field, iid_field)
    test_user_items = _extract_user_items(eval_data.dataset.inter_feat, uid_field, iid_field)

    rec_user_items = {}
    eval_users = sorted(test_user_items.keys())
    if not eval_users:
        return OrderedDict()

    users = torch.tensor(eval_users, dtype=torch.long, device=config["device"])
    item_count = dataset.item_num
    if batch_size is None:
        # RecBole's eval_batch_size is a score budget, not a user-count budget.
        effective_batch_size = max(score_budget // item_count, 1)
    else:
        effective_batch_size = max(int(batch_size), 1)
    effective_batch_size = min(effective_batch_size, len(users))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(users), effective_batch_size):
                batch_users = users[start:start + effective_batch_size]
                interaction = Interaction({uid_field: batch_users})
                scores = model.full_sort_predict(interaction).view(len(batch_users), item_count)
                scores[:, 0] = float("-inf")

                for row_idx, user_id in enumerate(batch_users.tolist()):
                    seen_items = train_user_items.get(int(user_id), [])
                    if seen_items:
                        scores[row_idx, seen_items] = float("-inf")

                _, topk_items = torch.topk(scores, k=k, dim=1)
                for user_id, items in zip(batch_users.tolist(), topk_items.cpu().tolist()):
                    rec_user_items[int(user_id)] = items
    finally:
        if was_training:
            model.train()

    return evaluate_fairness_from_topk(
        train_user_items=train_user_items,
        test_user_items=test_user_items,
        rec_user_items=rec_user_items,
        split_ratio=split_ratio,
        k=k,
    )


def evaluate_fairness_from_recbole_rec_file(
    model_path,
    rec_file,
    user_map_file,
    item_map_file,
    split_ratio=DEFAULT_SPLIT_RATIO,
    k=DEFAULT_K,
):
    load_data_and_model_flexible, _, normalize_fn = _require_predict_helpers()
    global normalize_org_id
    normalize_org_id = normalize_fn
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model_flexible(model_file=model_path)
    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]

    eval_data, eval_source = _select_eval_split(valid_data, test_data, uid_field)
    train_user_items = _extract_user_items(train_data.dataset.inter_feat, uid_field, iid_field)
    test_user_items = _extract_user_items(eval_data.dataset.inter_feat, uid_field, iid_field)
    if user_map_file and item_map_file:
        rec_user_items = _load_recbole_rec_file(
            rec_file=rec_file,
            user_map_file=user_map_file,
            item_map_file=item_map_file,
            dataset=dataset,
            uid_field=uid_field,
            iid_field=iid_field,
        )
    else:
        rec_user_items = _load_internal_rec_file(rec_file)

    result = evaluate_fairness_from_topk(
        train_user_items=train_user_items,
        test_user_items=test_user_items,
        rec_user_items=rec_user_items,
        split_ratio=split_ratio,
        k=k,
    )
    return OrderedDict([('eval_source', eval_source), *result.items()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec_file", required=True)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--user_map_file", default=None)
    parser.add_argument("--item_map_file", default=None)
    parser.add_argument("--split_ratio", type=float, default=DEFAULT_SPLIT_RATIO)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    if args.model_path:
        result = evaluate_fairness_from_recbole_rec_file(
            model_path=args.model_path,
            rec_file=args.rec_file,
            user_map_file=args.user_map_file,
            item_map_file=args.item_map_file,
            split_ratio=args.split_ratio,
            k=args.k,
        )
    else:
        if not args.train_file or not args.test_file:
            raise ValueError('train_file and test_file are required in file-based evaluation mode')
        train_user_items = _load_user_item_file(args.train_file)
        test_user_items = _load_user_item_file(args.test_file)
        rec_user_items = _load_user_item_file(args.rec_file)
        result = evaluate_fairness_from_topk(
            train_user_items=train_user_items,
            test_user_items=test_user_items,
            rec_user_items=rec_user_items,
            split_ratio=args.split_ratio,
            k=args.k,
        )

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
