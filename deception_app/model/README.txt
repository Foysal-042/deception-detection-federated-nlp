Place your trained federated global model checkpoint here as:

    global_banglishbert.pt

It must be a PyTorch state_dict saved from the DeceptionClassifier class
defined in app.py (encoder + attention pooling + classifier head), e.g.:

    torch.save(model.state_dict(), "global_banglishbert.pt")

where `model` is the final aggregated global model after the last FedProx
round in your Flower simulation.

If this file is missing, app.py will still start (for UI demo purposes)
but will print a warning and use an untrained classifier head.
