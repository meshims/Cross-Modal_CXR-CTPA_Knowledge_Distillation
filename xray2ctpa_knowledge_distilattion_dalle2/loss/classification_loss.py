import torch
import torch.nn as nn


MIN_VAL_FOR_DATASET=0 # TODO fill per dataset
MAX_VAL_FOR_DATASET=0 # TODO fill per dataset

def get_min_max_for_dataset():
    return MIN_VAL_FOR_DATASET, MAX_VAL_FOR_DATASET


def unNormalize(generated_ct):
    min_val, max_val = get_min_max_for_dataset()
    un_normalized_ct = (generated_ct*(max_val-min_val) + max_val +min_val)/2
    return un_normalized_ct

def load_classifier_model():
    net = # Load classification model
    net = net.cuda()
    return net

def classification_loss(predictions, target):
    classification_model = load_classifier_model()
    classification_model.eval()
    # Before using the classifier we need to unormalize the predictions as the classifier expects unormalized embeddings
    generated_cts = unNormalize(predictions)
    target_cts = unNormalize(target)
    bce_loss = nn.BCELoss()
    with torch.no_grad():
        generated_cts = generated_cts.cuda()
        target_cts = target_cts.cuda()
        classifier_generated_predictions = classification_model.classifier(generated_cts)
        classifier_generated_probabilities = torch.sigmoid(classifier_generated_predictions)
        classifier_target_predictions = classification_model.classifier(target_cts)
        classifier_target_probabilities = torch.sigmoid(classifier_target_predictions)

        loss = bce_loss(classifier_generated_probabilities, classifier_target_probabilities)


    return loss




