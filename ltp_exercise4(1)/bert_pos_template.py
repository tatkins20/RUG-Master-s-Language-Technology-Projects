# bert_pos.py
# author: Lukas Edman
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import os
from dataloading import POSDataset, padding_collate_fn, IDX2POS, POS2IDX, IGNORE_IDX
from torch.utils.data import DataLoader
from transformers import SqueezeBertConfig, SqueezeBertTokenizer, SqueezeBertForTokenClassification, DistilBertConfig, DistilBertTokenizer, DistilBertForTokenClassification

parser = argparse.ArgumentParser(description="POS tagging")
parser.add_argument("--reload_model", type=str, help="Path of model to reload")
parser.add_argument("--save_model", type=str, help="Path for saving the model")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_hidden_layers", type=int, default=1)
parser.add_argument("--num_attn_heads", type=int, default=1)
parser.add_argument("--output_file", type=str, help="Path for writing to a file")

"""
Answers to the questions:

- Q1. Fill in the code in the relevant functions below.
- Q2. Report here the accuracies on the dev set.

- Q3. Report here a description of the changes you've made, including the accuracy for each change on the devset. For each change also indicate where in the code (lines) the modification is made.

"""

# Q1c
def evaluate(model, loader):
    model.eval() # set the model into eval mode, disabling dropout
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch in loader:
            data, labels = batch
            # evaluation code here
            output = model(data, labels = labels)
            loss = output.loss
            logits = output.logits
            predictions = torch.argmax(logits, dim=2)
            #labelsIgn = labels[predictions==labels]
            #predictionsIgn = predictions[predictions==labels]
            total += (labels!=-100).sum().item()
            correct += (predictions==labels).sum().item()
            #print(total)
            #print(correct)
            #predictions = np.argmax(logits, axis=2)
            # Make sure to ignore the positions where IGNORE_IDX is in labels!

    model.train() # set the model back into training mode
    return correct/total

# Q1a and Q1b
def train(model, train_loader, valid_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.Rprop(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        running_loss = 0.0
        total_tokens = 0.0
        for i, batch in enumerate(train_loader):
            data, labels = batch
            total_tokens += data.numel()

            # i. zero gradients
            optimizer.zero_grad()
            # ii. do forward pass
            outputs = model(data, labels=labels)
            # iii. get loss
            loss = outputs.loss
            # add loss to total_loss
            total_loss += loss.item()
            running_loss += loss.item()
            if i % 5 == 0:    # print every 5 iterations
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/5))
            running_loss = 0.0
            # iv. do backward pass
            loss.backward()
            # v. take an optimization step
            optimizer.step()

            if i % 5 == 0 and i > 0:
                avg_loss = total_loss / 5.0
                toks_per_sec = total_tokens / (time.time()-start_time)
                print("[Epoch %d, Iter %d] loss: %.4f, toks/sec: %d" % \
                    (epoch, i, avg_loss, toks_per_sec))
                start_time = time.time()
                total_loss = 0.0
                total_tokens = 0.0


        acc = evaluate(model, valid_loader)
        print("[Epoch %d] Acc (valid): %.4f" % (epoch, acc))

    print("############## END OF TRAINING ##############")
    acc = evaluate(model, valid_loader)
    print("Final Acc (valid): %.4f" % (acc))

def write_to_file(model, loader, output_file):
    model.eval()
    # file output code here
    save_path = output_file
    name_of_file = "output_s4555376"
    completeName = os.path.join(save_path, name_of_file+".txt")
    #f = open(completeName, "w+")
    with open(completeName, 'w+') as f:
        with torch.no_grad():
            for batch in loader:
                data, labels = batch
                # evaluation code here
                output = model(data)
                loss = output.loss
                logits = output.logits
                predictions = torch.argmax(logits, dim=2)
                #labelsIgn = labels[predictions==labels]
                for i, sentence in enumerate(predictions):
                    for j, word in enumerate(sentence):
                        if predictions[i][j] != -100:
                            POS = IDX2POS[predictions[i][j]]
                            f.write(str(POS) + " ")
                #predictions = predictions.detach().numpy()
                    f.write("\n")
                #total += (labels!=-100).sum().item()
                #correct += (predictions==labels).sum().item()
    f.close()
    model.train()

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    args = parser.parse_args()

    # Load tokenizer

    pretrained = 'distilbert-base-uncased-distilled-squad'
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = False

    '''
    pretrained = 'squeezebert/squeezebert-uncased'
    tokenizer = SqueezeBertTokenizer.from_pretrained(pretrained)
    tokenizer.do_basic_tokenize = False
    '''

    # Load data
    train_dataset = POSDataset("data/train.en", "data/train.en.label", tokenizer)
    train_loader = DataLoader(train_dataset,
        shuffle=True,
        collate_fn=padding_collate_fn,
        batch_size=args.batch_size)
    valid_dataset = POSDataset("data/valid.en", "data/valid.en.label", tokenizer)
    valid_loader = DataLoader(valid_dataset,
        collate_fn=padding_collate_fn,
        batch_size=args.batch_size)
    test_dataset = POSDataset("data/test.en", None, tokenizer)
    test_loader = DataLoader(test_dataset,
        collate_fn=padding_collate_fn,
        batch_size=args.batch_size)

    # Configure model parameters
    config = SqueezeBertConfig.from_pretrained(pretrained)
    #config = DistilBertConfig.from_pretrained(pretrained)
    config.num_labels = len(IDX2POS)
    config.num_hidden_layers = args.num_hidden_layers
    config.num_attention_heads = args.num_attn_heads
    #config.hidden_dropout_prob = 0.2
    config.output_attentions = True
    #config.attention_probs_dropout_prob = 0.2
    # Load an untrained model
    #model = SqueezeBertForTokenClassification(config)

    # Load a pretrained model
    '''
    model = SqueezeBertForTokenClassification.from_pretrained(
         pretrained,
         num_labels=len(IDX2POS),
         num_hidden_layers=args.num_hidden_layers,
         num_attention_heads=args.num_attn_heads,
         output_attentions=True)
    '''
    model = DistilBertForTokenClassification.from_pretrained(
         pretrained,
         num_labels=len(IDX2POS),
         output_attentions=True)

    '''
    Q2.
    Be sure to report the accuracies for each experiment and whether the results are expected or not.

    a) Compare an untrained model versus a pretrained model.
    Untrained model accuracy: 0.7822
    Pretrained model accuracy: 0.8756
    Expectations: The results are as expected that the pretrained model has performed with higher
    accuracy than the untrained model.

    b) Scale up the model by increasing it to 2 layers. Do this on the pretrained model.
    Scaled up pretrained model - (num_hidden_layers = 2) accuracy: 0.9027
    Expecations: The results are as expected that adding a few more layers to the pretrained model
    increases the accuracy performance.

    3) Changed model to DistilBERT and pretrained model to uncased base DistilBERT
    DistilBert Model accuracy: 0.9650
    '''
    '''
    model = SqueezeBertForTokenClassification.from_pretrained(
    pretrained,
    num_labels=len(IDX2POS),
    num_hidden_layers=args.num_hidden_layers,
    num_attention_heads=args.num_attn_heads,
    output_attentions=True
    )
    '''

    # Load model weights from a file
    if args.reload_model:
        model.load_state_dict(torch.load(args.reload_model))

    # Train
    train(model, train_loader, valid_loader, args.epochs)

    acc = evaluate(model, valid_loader)
    print("Acc (valid): %.4f" % (acc))
    # Write output to file
    if args.output_file:
        write_to_file(model, test_loader, args.output_file)

    # Save model weights
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
