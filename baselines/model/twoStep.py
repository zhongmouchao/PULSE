import copy
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)


def twoStep(
    X_observed,
    X_unobserved,
    X_test,
    final_classifier,
    reliable_classifier,
    spy_seed,
    device=None,
    penalty_coeff=50,
):
    """
    Two-step PU classification (pure torch version)

    Input:
        X_observed: observed data (torch.Tensor)
        X_unobserved: unlabeled data (torch.Tensor)
        X_test: test data (torch.Tensor)
        final_classifier: final classifier
        reliable_classifier: reliable classifier
        spy_seed: seed
        device: device
    Output:
        final_predictions: torch.Tensor
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # move tensors to device. Each test matrix is scored after the same two fits.
    several_tests = isinstance(X_test, (list, tuple))
    test_matrices = list(X_test) if several_tests else [X_test]
    X_observed = torch.as_tensor(X_observed, device=device, dtype=torch.float32)
    X_unobserved = torch.as_tensor(X_unobserved, device=device, dtype=torch.float32)

    # =========================================================
    # Shared optimization setup
    # =========================================================
    base_lr = 1e-3
    max_lr = 1e-1
    step_size_up = 50
    max_epochs = 2000
    patience = 100
    min_delta = 1e-6
    max_grad_norm = 1.0

    # =========================================================
    # Helper: batched prediction
    # =========================================================
    def predict_proba(model, X_tensor):
        model.eval()

        with torch.no_grad():
            out = model(X_tensor)
    
        return out

    # =========================================================
    # Helper: shared training routine
    # =========================================================
    def train_classifier(model, X_train, y_train, stage_name="training", penalty_coeff=50):
        model = model.to(device)
        model.train()

        y_train = y_train.view(-1, 1).float()

        criterion = nn.BCELoss()

        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode="triangular",
            cycle_momentum=False,
        )

        best_loss = float("inf")
        best_model = None
        patience_counter = 0

        for epoch in range(max_epochs):
            optimizer.zero_grad(set_to_none=True)

            out = model(X_train)
            loss = criterion(out.float(), y_train.float())

            class_param = torch.cat([p.view(-1) for p in model.parameters()])
            loss = loss + penalty_coeff * torch.sum(torch.square(class_param))

            if not torch.isfinite(loss):
                logging.info(f"Non-finite loss at epoch {epoch} during {stage_name}.")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            current_loss = loss.item()

            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logging.info(
                    f"{stage_name} | Epoch {epoch}, Loss: {current_loss:.6f}, LR: {current_lr:.6f}"
                )

            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch} during {stage_name}")
                break

        if best_model is None:
            raise RuntimeError(f"{stage_name} failed: no valid model found.")

        return best_model

    # =========================================================
    # Seed
    # =========================================================
    torch.manual_seed(spy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spy_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # =========================================================
    # Step 1: Spy selection
    # =========================================================
    n_observed = X_observed.shape[0]
    n_spy = int(0.2 * n_observed)

    perm = torch.randperm(n_observed, device=device)
    spy_indices = perm[:n_spy]
    positive_indices = perm[n_spy:]

    spy_data = X_observed[spy_indices]
    positive_data = X_observed[positive_indices]

    negative_data = torch.cat((spy_data, X_unobserved), dim=0)

    # =========================================================
    # Step 2: Train reliable classifier
    # =========================================================
    X_train_stage1 = torch.cat((positive_data, negative_data), dim=0)
    y_train_stage1 = torch.cat((
        torch.ones(len(positive_data), device=device, dtype=torch.float32),
        torch.zeros(len(negative_data), device=device, dtype=torch.float32),
    ), dim=0)

    reliable_classifier = train_classifier(
        model=reliable_classifier,
        X_train=X_train_stage1,
        y_train=y_train_stage1,
        stage_name="reliable classifier training",
        penalty_coeff=penalty_coeff,
    )

    # =========================================================
    # Step 3: Find reliable negatives
    # =========================================================
    spy_predictions = predict_proba(
        reliable_classifier, spy_data
    )
    unlabeled_predictions = predict_proba(
        reliable_classifier, X_unobserved
    )

    threshold = torch.min(spy_predictions)
    logging.info(f"threshold for reliable negatives: {threshold.item()}")

    reliable_negatives_indices = torch.where(unlabeled_predictions < threshold)[0]
    reliable_negatives = X_unobserved[reliable_negatives_indices]
    logging.info(f"Number of reliable negatives: {len(reliable_negatives)}")

    # avoid case where no reliable negatives are found
    if reliable_negatives.shape[0] == 0:
        logging.info("No reliable negatives found. Falling back to training on observed only.")
        reliable_negatives = torch.empty(
            (0, X_observed.shape[1]),
            dtype=X_observed.dtype,
            device=device,
        )

    # =========================================================
    # Step 4: Train final classifier
    # =========================================================
    X_train_stage2 = torch.cat((X_observed, reliable_negatives), dim=0)
    y_train_stage2 = torch.cat((
        torch.ones(len(X_observed), device=device, dtype=torch.float32),
        torch.zeros(len(reliable_negatives), device=device, dtype=torch.float32),
    ), dim=0)

    final_classifier = train_classifier(
        model=final_classifier,
        X_train=X_train_stage2,
        y_train=y_train_stage2,
        stage_name="final classifier training",
        penalty_coeff=penalty_coeff,
    )

    # =========================================================
    # Step 5: Predict on each test file
    # =========================================================
    predictions = []
    for features in test_matrices:
        test_tensor = torch.as_tensor(features, device=device, dtype=torch.float32)
        predictions.append(predict_proba(final_classifier, test_tensor).cpu().detach().numpy().reshape(-1))
    if several_tests:
        return predictions
    return predictions[0]